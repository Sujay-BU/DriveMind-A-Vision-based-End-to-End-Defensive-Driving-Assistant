"""DriveTransformer's three attention operations, plus the safety-query wiring.

Each block runs, in order:

1. **Sensor cross-attention** -- every task query attends to the multi-view
   image tokens, so planning touches raw pixels directly rather than through a
   perception bottleneck.
2. **Task self-attention** -- agent, map, planning *and safety* queries attend
   to each other.  This is where thesis modification 1 does its work: "the
   agent, planning and map queries attend to the safety query directly at every
   block of the transformer architecture."
3. **Temporal cross-attention** -- queries attend to a FIFO of past query
   states to recover intent.

The paper's key claim is that these three run in *parallel* over a shared query
set rather than as a sequential perception -> prediction -> planning pipeline,
which is what keeps training stable as the model scales.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualAttention(nn.Module):
    """Pre-norm multi-head attention with a residual connection."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(query)
        kv = q if key_value is None else self.norm_kv(key_value)
        out, _ = self.attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        return query + out


class DriveTransformerBlock(nn.Module):
    """One unified block over the concatenated query set.

    The query set is laid out as ``[agent | map | planning | safety]`` along the
    sequence axis.  Keeping them in one tensor is what makes task
    self-attention a single operation and lets the safety query be attended to
    by all three original query types for free.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.sensor_attn = ResidualAttention(dim, num_heads, dropout)
        self.task_attn = ResidualAttention(dim, num_heads, dropout)
        self.use_temporal = use_temporal
        if use_temporal:
            self.temporal_attn = ResidualAttention(dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio, dropout)

    def forward(
        self,
        queries: torch.Tensor,
        sensor_tokens: torch.Tensor,
        temporal_tokens: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        queries = self.sensor_attn(queries, sensor_tokens)
        queries = self.task_attn(queries)
        if self.use_temporal and temporal_tokens is not None:
            queries = self.temporal_attn(
                queries, temporal_tokens, key_padding_mask=temporal_mask
            )
        return queries + self.ffn(self.ffn_norm(queries))


class QuerySet:
    """Bookkeeping for the four query groups packed into one tensor.

    Slicing by name keeps the block code free of magic index arithmetic, which
    matters because adding the safety query shifts every downstream offset.
    """

    def __init__(self, num_agent: int, num_map: int, num_plan: int, num_safety: int = 1):
        self.num_agent = num_agent
        self.num_map = num_map
        self.num_plan = num_plan
        self.num_safety = num_safety

    @property
    def total(self) -> int:
        return self.num_agent + self.num_map + self.num_plan + self.num_safety

    @property
    def agent_slice(self) -> slice:
        return slice(0, self.num_agent)

    @property
    def map_slice(self) -> slice:
        return slice(self.num_agent, self.num_agent + self.num_map)

    @property
    def plan_slice(self) -> slice:
        start = self.num_agent + self.num_map
        return slice(start, start + self.num_plan)

    @property
    def safety_slice(self) -> slice:
        start = self.num_agent + self.num_map + self.num_plan
        return slice(start, start + self.num_safety)

    def split(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "agent": packed[:, self.agent_slice],
            "map": packed[:, self.map_slice],
            "plan": packed[:, self.plan_slice],
            "safety": packed[:, self.safety_slice],
        }
