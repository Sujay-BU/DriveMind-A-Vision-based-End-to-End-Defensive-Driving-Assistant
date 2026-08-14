"""Multi-view image backbone producing the sensor token sequence.

DriveTransformer's sensor cross-attention lets every task query attend to raw
image features, so the backbone's job is to turn ``[B, N_cam, 3, H, W]`` into a
flat token sequence ``[B, N_cam * H' * W', D]`` carrying enough positional and
per-camera identity information for attention to be geometrically meaningful.
"""

from __future__ import annotations

import math
from typing import List

import timm
import torch
import torch.nn as nn
import torch.utils.checkpoint


def sinusoid_2d(height: int, width: int, dim: int, device=None) -> torch.Tensor:
    """Fixed 2D sine-cosine position embedding, ``[H*W, dim]``.

    Half the channels encode the row, half the column.  Fixed rather than
    learned so that a model trained at one feature-map resolution still makes
    sense if the input resolution changes between the tiny and large configs.
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sincos, got {dim}")
    quarter = dim // 4
    omega = torch.exp(
        torch.arange(quarter, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / max(quarter - 1, 1))
    )
    ys = torch.arange(height, dtype=torch.float32, device=device)
    xs = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    def embed(coords: torch.Tensor) -> torch.Tensor:
        out = coords.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([out.sin(), out.cos()], dim=1)

    return torch.cat([embed(grid_y), embed(grid_x)], dim=1)


class MultiViewBackbone(nn.Module):
    """Shared-weight 2D backbone applied to every camera view.

    Parameters
    ----------
    name
        Any ``timm`` model name. ``resnet18``/``resnet34`` for the 6 GB config,
        ``resnet101``/``vit_large_patch14_dinov2`` for the cluster config.
    num_cameras
        Number of views. The thesis uses 6 (front, front-left, front-right,
        rear, rear-left, rear-right).
    out_dim
        Model width D that sensor tokens are projected to.
    out_indices
        Which backbone stages to take features from. Multiple stages give the
        multi-scale behaviour TransFuser/DriveTransformer rely on; the tokens
        from each stage are concatenated along the sequence axis.
    """

    def __init__(
        self,
        name: str = "resnet34",
        num_cameras: int = 6,
        out_dim: int = 256,
        out_indices: tuple[int, ...] = (2, 3),
        pretrained: bool = True,
        freeze_stem: bool = False,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.out_dim = out_dim

        self.body = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        channels: List[int] = self.body.feature_info.channels()
        self.projections = nn.ModuleList(
            [nn.Conv2d(c, out_dim, kernel_size=1) for c in channels]
        )
        # One learnable embedding per camera, so attention can tell a rear view
        # from a front view even where the pixels look alike.
        self.camera_embedding = nn.Parameter(torch.zeros(num_cameras, out_dim))
        nn.init.normal_(self.camera_embedding, std=0.02)
        # One per scale, for the same reason across stages.
        self.scale_embedding = nn.Parameter(torch.zeros(len(channels), out_dim))
        nn.init.normal_(self.scale_embedding, std=0.02)

        self.norm = nn.LayerNorm(out_dim)
        self._pos_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

        # Checkpoint the backbone through torch directly rather than timm's
        # ``set_grad_checkpointing``. timm uses the reentrant implementation,
        # which cannot recompute a graph whose first op is an inplace ReLU on a
        # leaf tensor -- exactly the shape of a ResNet stem -- and raises
        # "a leaf Variable that requires grad is being used in an in-place
        # operation" on the backward pass.
        self.grad_checkpointing = grad_checkpointing
        if freeze_stem:
            for p in list(self.body.parameters())[:2]:
                p.requires_grad_(False)

    def _pos(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device)
        if key not in self._pos_cache:
            self._pos_cache[key] = sinusoid_2d(h, w, self.out_dim, device=device)
        return self._pos_cache[key]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, N_cam, 3, H, W]`` -> ``[B, L, D]`` sensor tokens."""
        b, n, c, h, w = images.shape
        if n != self.num_cameras:
            raise ValueError(f"expected {self.num_cameras} cameras, got {n}")

        flat = images.reshape(b * n, c, h, w)
        if self.grad_checkpointing and self.training:
            feats = torch.utils.checkpoint.checkpoint(
                self.body, flat, use_reentrant=False
            )
        else:
            feats = self.body(flat)

        tokens = []
        for scale_idx, (feat, proj) in enumerate(zip(feats, self.projections)):
            x = proj(feat)  # [B*N, D, h', w']
            _, d, fh, fw = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B*N, h'w', D]
            x = x + self._pos(fh, fw, x.device).unsqueeze(0)
            x = x + self.scale_embedding[scale_idx].reshape(1, 1, d)
            x = x.reshape(b, n, fh * fw, d)
            x = x + self.camera_embedding.reshape(1, n, 1, d)
            tokens.append(x.reshape(b, n * fh * fw, d))

        return self.norm(torch.cat(tokens, dim=1))
