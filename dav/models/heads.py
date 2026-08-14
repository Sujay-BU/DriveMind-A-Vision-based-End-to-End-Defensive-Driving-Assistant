"""Output heads: agent (detection + motion), map, and the planning head.

Thesis modifications 2 and 4 live here, because both change existing
DriveTransformer components rather than adding new ones.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

#: The six trajectory modes the thesis names for the planning head.
TRAJECTORY_MODES = ("straight", "stop", "left", "sharp_left", "right", "sharp_right")
NUM_MODES = len(TRAJECTORY_MODES)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class AgentHead(nn.Module):
    """Per-agent 3D box, class, and future motion.

    Boxes are ``(x, y, z, w, l, h, sin_yaw, cos_yaw)`` in the ego frame; yaw is
    predicted as a sin/cos pair so the loss has no wrap-around discontinuity.
    """

    BOX_DIM = 8

    def __init__(
        self, dim: int, num_classes: int, motion_steps: int = 6, hidden: Optional[int] = None
    ) -> None:
        super().__init__()
        hidden = hidden or dim
        self.motion_steps = motion_steps
        self.cls = mlp(dim, hidden, num_classes + 1)  # +1 for "no object"
        self.box = mlp(dim, hidden, self.BOX_DIM)
        self.motion = mlp(dim, hidden, motion_steps * 2)

    def forward(self, agent_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n, _ = agent_queries.shape
        return {
            "agent_logits": self.cls(agent_queries),
            "agent_boxes": self.box(agent_queries),
            "agent_motion": self.motion(agent_queries).reshape(b, n, self.motion_steps, 2),
        }


class MapHead(nn.Module):
    """Vectorised map elements: polyline points plus an element class."""

    def __init__(
        self,
        dim: int,
        num_classes: int = 3,  # lane divider, road boundary, pedestrian crossing
        num_points: int = 20,
        hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden = hidden or dim
        self.num_points = num_points
        self.cls = mlp(dim, hidden, num_classes + 1)
        self.points = mlp(dim, hidden, num_points * 2)

    def forward(self, map_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n, _ = map_queries.shape
        return {
            "map_logits": self.cls(map_queries),
            "map_points": self.points(map_queries).reshape(b, n, self.num_points, 2),
        }


class SafetyAwarePlanningPE(nn.Module):
    """Thesis modification 2: safety-conditioned planning positional encoding.

        PE_ego(l+1) = MLP(predicted_trajectory(l) (+) safety_query(timestep))

    The formula has no value at ``l = 0``: there is no ``predicted_trajectory``
    before the first block has run.  A learnable trajectory prior stands in for
    it, which is the same device DETR-style decoders use for their initial
    reference points.  See DOCUMENTATION.md, deviation D8.
    """

    def __init__(
        self, dim: int, horizon: int, hidden: Optional[int] = None
    ) -> None:
        super().__init__()
        hidden = hidden or dim
        self.horizon = horizon
        self.trajectory_prior = nn.Parameter(torch.zeros(horizon, 2))
        self.mlp = mlp(horizon * 2 + dim, hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        safety_token: torch.Tensor,
        trajectory: Optional[torch.Tensor] = None,
        batch: Optional[int] = None,
    ) -> torch.Tensor:
        """``[B, 1, D]`` safety token (+ optional ``[B, T, 2]`` trajectory) -> ``[B, 1, D]``."""
        if safety_token.dim() == 3:
            safety_token = safety_token.squeeze(1)
        b = batch or safety_token.shape[0]

        if trajectory is None:
            traj = self.trajectory_prior.unsqueeze(0).expand(b, -1, -1)
        else:
            traj = trajectory
        traj_flat = traj.reshape(b, -1)

        return self.norm(self.mlp(torch.cat([traj_flat, safety_token], dim=-1))).unsqueeze(1)


class SafetyAwarePlanningHead(nn.Module):
    """Thesis modification 4: safety-conditioned multi-modal planning head.

        trajectory_modes = MLP(H_ego (+) H_safety (+) mode_embeddings)

    Produces one trajectory per mode plus a score over modes, so the model can
    "learn which trajectory mode out of the 6 is the most consistent with the
    defensive driving metric's thresholds".

    Also emits a per-mode predicted safety ratio vector.  That is not in the
    thesis, but without it mode selection has no direct safety signal -- the
    mode scores would be trained only through the trajectory regression, and
    the stated goal is explicitly to pick the mode most consistent with the
    thresholds.  See DOCUMENTATION.md, deviation D10.
    """

    def __init__(
        self,
        dim: int,
        horizon: int = 6,
        num_modes: int = NUM_MODES,
        num_safety_metrics: int = 6,
        hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden = hidden or dim * 2
        self.horizon = horizon
        self.num_modes = num_modes

        self.mode_embeddings = nn.Parameter(torch.zeros(num_modes, dim))
        nn.init.normal_(self.mode_embeddings, std=0.02)

        fused = dim * 3  # H_ego (+) H_safety (+) mode_embedding
        self.trajectory = mlp(fused, hidden, horizon * 2, layers=3)
        self.mode_score = mlp(fused, hidden, 1, layers=2)
        self.mode_safety = mlp(fused, hidden, num_safety_metrics, layers=2)

    def forward(
        self, ego_query: torch.Tensor, safety_token: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        ego_query : ``[B, 1, D]`` (or ``[B, D]``) planning query.
        safety_token : ``[B, 1, D]`` (or ``[B, D]``) final safety query.
        """
        if ego_query.dim() == 3:
            ego_query = ego_query.mean(dim=1)
        if safety_token.dim() == 3:
            safety_token = safety_token.squeeze(1)

        b, d = ego_query.shape
        m = self.num_modes

        ego = ego_query.unsqueeze(1).expand(b, m, d)
        safety = safety_token.unsqueeze(1).expand(b, m, d)
        modes = self.mode_embeddings.unsqueeze(0).expand(b, m, d)
        fused = torch.cat([ego, safety, modes], dim=-1)

        return {
            "mode_trajectories": self.trajectory(fused).reshape(b, m, self.horizon, 2),
            "mode_scores": self.mode_score(fused).squeeze(-1),
            "mode_safety_ratios": nn.functional.softplus(self.mode_safety(fused) + 1.0),
        }

    @staticmethod
    def select(outputs: Dict[str, torch.Tensor], safety_gate: bool = True) -> torch.Tensor:
        """Pick one trajectory per sample from the mode set.

        With ``safety_gate`` on, modes whose predicted worst ratio exceeds 1.0
        are pushed down before the argmax, so a high-scoring but
        threshold-violating mode loses to a compliant one.  This is the
        inference-time expression of "the most consistent with the defensive
        driving metric's thresholds"; it is a soft penalty rather than a hard
        mask so that a sample with *no* compliant mode still yields the least
        bad trajectory instead of nothing.
        """
        scores = outputs["mode_scores"]
        if safety_gate:
            worst = outputs["mode_safety_ratios"].amax(dim=-1)  # [B, M]
            scores = scores - 10.0 * torch.relu(worst - 1.0)
        best = scores.argmax(dim=-1)  # [B]
        traj = outputs["mode_trajectories"]
        index = best.reshape(-1, 1, 1, 1).expand(-1, 1, traj.shape[2], traj.shape[3])
        return traj.gather(1, index).squeeze(1)
