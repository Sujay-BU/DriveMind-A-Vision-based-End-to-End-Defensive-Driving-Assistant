"""Safety-aware DriveTransformer -- the model of the thesis.

This is a re-implementation of DriveTransformer (Jia et al., 2025) with the
five safety modifications from Chapter 3 applied.  It is a re-implementation
rather than a patch on the authors' repository so that the tiny/large configs
share one code path and the whole thing runs without an mmdet3d install; the
architectural contract (query-based, three parallel attention operations,
FIFO temporal memory) follows the paper.

Where each thesis modification lives:

  1. safety query as a fourth query type      -> ``SafetyQueryEncoder`` + ``QuerySet``
  2. planning positional encoding concatenation -> ``SafetyAwarePlanningPE``
  3. safety Huber loss on 6 predicted metrics -> ``SafetyPredictionHead`` + ``SafetyHuberLoss``
  4. safety-conditioned planning head          -> ``SafetyAwarePlanningHead``
  5. safety temporal cross-attention           -> ``SafetyTemporalMemory``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ..metrics.thresholds import NUM_SAFETY_METRICS
from .attention import DriveTransformerBlock, QuerySet
from .backbone import MultiViewBackbone
from .heads import (
    NUM_MODES,
    AgentHead,
    MapHead,
    SafetyAwarePlanningHead,
    SafetyAwarePlanningPE,
)
from .safety import SafetyPredictionHead, SafetyQueryEncoder, SafetyTemporalMemory


@dataclass
class DAVConfig:
    """Model configuration. See ``configs/*.yaml`` for the two shipped presets."""

    # Backbone
    backbone: str = "resnet34"
    pretrained: bool = True
    num_cameras: int = 6
    image_size: tuple[int, int] = (224, 400)
    backbone_out_indices: tuple[int, ...] = (2, 3)

    # Transformer
    dim: int = 256
    num_blocks: int = 6
    num_heads: int = 8
    ffn_ratio: float = 4.0
    dropout: float = 0.0

    # Queries
    num_agent_queries: int = 128
    num_map_queries: int = 64
    num_plan_queries: int = 1

    # Outputs
    num_agent_classes: int = 10
    num_map_classes: int = 3
    map_points: int = 20
    motion_steps: int = 6
    horizon: int = 6  # planning waypoints

    # Temporal
    temporal_length: int = 4  # N in the FIFO queue
    use_temporal: bool = True

    # Safety modifications. Each can be switched off independently, which is
    # what makes the ablation table in the thesis reproducible.
    use_safety_query: bool = True
    use_safety_pe: bool = True
    use_safety_planning_head: bool = True
    use_safety_temporal: bool = True

    # Efficiency
    grad_checkpointing: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)
        if isinstance(self.backbone_out_indices, list):
            self.backbone_out_indices = tuple(self.backbone_out_indices)


class SafetyAwareDriveTransformer(nn.Module):
    def __init__(self, cfg: DAVConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.dim

        self.backbone = MultiViewBackbone(
            name=cfg.backbone,
            num_cameras=cfg.num_cameras,
            out_dim=d,
            out_indices=cfg.backbone_out_indices,
            pretrained=cfg.pretrained,
            grad_checkpointing=cfg.grad_checkpointing,
        )

        num_safety = 1 if cfg.use_safety_query else 0
        self.queries = QuerySet(
            cfg.num_agent_queries, cfg.num_map_queries, cfg.num_plan_queries, num_safety
        )

        # Learnable initial content for the three original query types. The
        # safety query is *not* learnable-from-scratch: per the thesis it is
        # built from scalar metric data on every forward pass.
        self.agent_queries = nn.Parameter(torch.zeros(cfg.num_agent_queries, d))
        self.map_queries = nn.Parameter(torch.zeros(cfg.num_map_queries, d))
        self.plan_queries = nn.Parameter(torch.zeros(cfg.num_plan_queries, d))
        for p in (self.agent_queries, self.map_queries, self.plan_queries):
            nn.init.normal_(p, std=0.02)

        # Ego status conditioning, as in UniAD and the DriveTransformer paper:
        # speed, acceleration, steering, and the high-level navigation command.
        self.ego_state_encoder = nn.Sequential(
            nn.Linear(6, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )
        self.command_embedding = nn.Embedding(7, d)  # CARLA has 6 commands + unknown

        if cfg.use_safety_query:
            self.safety_encoder = SafetyQueryEncoder(d, NUM_SAFETY_METRICS)
            self.safety_head = SafetyPredictionHead(d, NUM_SAFETY_METRICS)
        if cfg.use_safety_pe:
            self.safety_pe = SafetyAwarePlanningPE(d, cfg.horizon)
        if cfg.use_safety_temporal:
            self.safety_temporal = SafetyTemporalMemory(d, cfg.num_heads, cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                DriveTransformerBlock(
                    d, cfg.num_heads, cfg.ffn_ratio, cfg.dropout, cfg.use_temporal
                )
                for _ in range(cfg.num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d)

        self.agent_head = AgentHead(d, cfg.num_agent_classes, cfg.motion_steps)
        self.map_head = MapHead(d, cfg.num_map_classes, cfg.map_points)

        # One head class for both settings. With the modification disabled the
        # head is fed a zero safety token in ``forward``, which reduces it to
        # the baseline "ego query (+) mode embeddings" form and keeps the
        # parameter count identical across the ablation.
        self.planning_head = SafetyAwarePlanningHead(
            d, cfg.horizon, NUM_MODES, NUM_SAFETY_METRICS
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        ego_state: torch.Tensor,
        command: torch.Tensor,
        safety_ratios: Optional[torch.Tensor] = None,
        safety_mask: Optional[torch.Tensor] = None,
        safety_valid: Optional[torch.Tensor] = None,
        safety_history: Optional[torch.Tensor] = None,
        safety_history_mask: Optional[torch.Tensor] = None,
        safety_history_ratios: Optional[torch.Tensor] = None,
        safety_history_masks: Optional[torch.Tensor] = None,
        temporal_tokens: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : ``[B, N_cam, 3, H, W]``
        ego_state : ``[B, 6]`` -- speed, lon accel, lat accel, steer, lon jerk, lat jerk
        command : ``[B]`` int64 high-level navigation command
        safety_ratios : ``[B, 6]`` previous-timestep violation ratios (see D5)
        safety_mask : ``[B, 6]`` validity per metric
        safety_valid : ``[B]`` 0 on the first frame of an episode
        safety_history : ``[B, N, D]`` past safety embeddings for modification 5
        safety_history_ratios : ``[B, N, 6]`` past ratio vectors. Preferred over
            ``safety_history``: they are re-encoded here by the current safety
            encoder, so the history never goes stale as the weights move and it
            carries gradient. See D11. Ignored if ``safety_history`` is given.
        temporal_tokens : ``[B, M, D]`` DriveTransformer's own FIFO of scene queries
        """
        cfg = self.cfg
        b = images.shape[0]
        device = images.device

        sensor_tokens = self.backbone(images)

        ego_embedding = self.ego_state_encoder(ego_state) + self.command_embedding(command)

        parts = [
            self.agent_queries.unsqueeze(0).expand(b, -1, -1),
            self.map_queries.unsqueeze(0).expand(b, -1, -1),
            # The planning query carries ego status, as in the base architecture.
            self.plan_queries.unsqueeze(0).expand(b, -1, -1) + ego_embedding.unsqueeze(1),
        ]

        safety_token: Optional[torch.Tensor] = None
        if cfg.use_safety_query:
            if safety_ratios is None:
                safety_ratios = torch.zeros(b, NUM_SAFETY_METRICS, device=device)
                safety_mask = torch.zeros(b, NUM_SAFETY_METRICS, device=device)
                safety_valid = torch.zeros(b, device=device)
            safety_token = self.safety_encoder(safety_ratios, safety_mask, safety_valid)

            # Modification 2: fold the safety token into the planning query's
            # positional encoding before the first block.
            if cfg.use_safety_pe:
                parts[2] = parts[2] + self.safety_pe(safety_token, batch=b)

            parts.append(safety_token)

        queries = torch.cat(parts, dim=1)

        for block in self.blocks:
            if cfg.grad_checkpointing and self.training:
                queries = checkpoint.checkpoint(
                    block,
                    queries,
                    sensor_tokens,
                    temporal_tokens,
                    temporal_mask,
                    use_reentrant=False,
                )
            else:
                queries = block(queries, sensor_tokens, temporal_tokens, temporal_mask)

        queries = self.final_norm(queries)
        split = self.queries.split(queries)

        ego_query = split["plan"]
        final_safety = split["safety"] if cfg.use_safety_query else None

        # Modification 5: the ego query attends to the history of safety states.
        if cfg.use_safety_temporal and cfg.use_safety_query:
            history = safety_history
            if history is None and safety_history_ratios is not None:
                # Re-encode the past N ratio vectors with the current encoder.
                bh, n, k = safety_history_ratios.shape
                flat_ratios = safety_history_ratios.reshape(bh * n, k)
                flat_masks = (
                    safety_history_masks.reshape(bh * n, k)
                    if safety_history_masks is not None
                    else None
                )
                history = self.safety_encoder(flat_ratios, flat_masks).reshape(bh, n, cfg.dim)
            ego_query = self.safety_temporal(ego_query, history, safety_history_mask)

        outputs: Dict[str, torch.Tensor] = {}
        outputs.update(self.agent_head(split["agent"]))
        outputs.update(self.map_head(split["map"]))

        # Modification 4: planning head conditioned on the safety token.
        if final_safety is not None and cfg.use_safety_planning_head:
            head_safety = final_safety
        else:
            head_safety = torch.zeros(b, 1, cfg.dim, device=device)
        outputs.update(self.planning_head(ego_query, head_safety))

        outputs["trajectory"] = SafetyAwarePlanningHead.select(
            outputs, safety_gate=cfg.use_safety_planning_head
        )

        # Modification 3: regress the six metrics from the final safety query.
        if final_safety is not None:
            outputs["safety_query"] = final_safety
            outputs["pred_safety_ratios"] = self.safety_head(final_safety)

        outputs["ego_query"] = ego_query
        return outputs

    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        if trainable_only:
            ps = (p for p in ps if p.requires_grad)
        return sum(p.numel() for p in ps)


def build_model(cfg: DAVConfig | dict) -> SafetyAwareDriveTransformer:
    if isinstance(cfg, dict):
        cfg = DAVConfig(**cfg)
    return SafetyAwareDriveTransformer(cfg)
