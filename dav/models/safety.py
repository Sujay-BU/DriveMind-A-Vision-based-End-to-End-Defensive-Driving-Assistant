"""The safety query: encoder, prediction head, and temporal safety memory.

This module holds thesis modifications 1, 3 and 5.  Modifications 2 (planning
positional encoding) and 4 (planning head) live in ``heads.py`` because they
alter existing DriveTransformer components rather than adding new ones.

Two design decisions here depart from the literal text; both are recorded in
DOCUMENTATION.md as D5 and D6.

**D5 -- where the safety scalars come from at inference.**  The thesis says the
safety query "is initialized at the beginning of every forward pass using the
data derived from the dataset collected", i.e. from ground-truth per-frame
metric scalars stored alongside the training data.  Those scalars do not exist
at test time: in closed-loop CARLA evaluation there is no annotation file to
read.  Conditioning on them during training and having nothing to condition on
during evaluation is straightforward train/test leakage -- the model would
learn to lean on an input that vanishes exactly when it matters.

Five of the six scalars are pure ego state (longitudinal and lateral
acceleration, longitudinal and lateral jerk, speed) and *are* available at
inference from the vehicle's own IMU and speedometer, one timestep in arrears.
Only the following distance needs perception.  So the encoder takes:

    - the five ego-state ratios, measured from the previous timestep (causally
      available both in training and in closed loop), and
    - the following-distance ratio, which at training time is the measured
      value and at inference time is the model's own previous-step estimate,
      with a validity flag marking which of the two it is.

Scheduled sampling anneals training from measured to self-estimated following
distance, so the model is never surprised by the switch at deployment.

**D6 -- the [0, 1] normalisation.**  The thesis says "all the dimensions of the
safety token are normalized between [0, 1] ... any token value produced during
evaluation or test time that exceeds 1 after normalization is considered
unsafe".  These two sentences contradict each other: a value squashed into
[0, 1] can never exceed 1.  The intent is clear from context -- 1.0 should mean
"exactly at threshold" -- so the *input ratios* carry that semantics (value /
threshold, unbounded above, see ``dav.metrics.evaluator.safety_vector``) while
the *token* is an ordinary unconstrained embedding.  Safety is read out through
the prediction head, which regresses ratios in the same unbounded space, not by
inspecting raw token dimensions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..metrics.thresholds import NUM_SAFETY_METRICS


class SafetyQueryEncoder(nn.Module):
    """Thesis modification 1: build the safety token from scalar metric data.

    "This scalar metric data is passed through a 2-layer MLP to obtain the
    safety token."

    Input is the six violation ratios plus their six validity flags, so the
    network can distinguish "following gap is 0.0 because the road ahead is
    clear" from "following gap ratio is 0.0", which are very different states.
    """

    def __init__(
        self,
        dim: int,
        num_metrics: int = NUM_SAFETY_METRICS,
        hidden: Optional[int] = None,
        ratio_clip: float = 5.0,
    ) -> None:
        super().__init__()
        hidden = hidden or dim
        self.num_metrics = num_metrics
        # Ratios are unbounded above; a single catastrophic frame (emergency
        # brake at 20x threshold) would otherwise dominate the input scale.
        self.ratio_clip = ratio_clip

        self.mlp = nn.Sequential(
            nn.Linear(num_metrics * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)
        # Fallback content for the very first frame of an episode, where no
        # previous-timestep ego state exists yet.
        self.null_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.null_token, std=0.02)

    def forward(
        self,
        ratios: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        valid_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[B, 6]`` ratios -> ``[B, 1, D]`` safety query.

        Parameters
        ----------
        ratios
            Violation ratios, 1.0 == at threshold.
        mask
            ``[B, 6]``, 1.0 where the metric is defined this frame.
        valid_frame
            ``[B]``, 0.0 for the first frame of an episode, where the whole
            vector is undefined and the null token is substituted.
        """
        if mask is None:
            mask = torch.ones_like(ratios)
        clipped = ratios.clamp(0.0, self.ratio_clip) * mask
        token = self.mlp(torch.cat([clipped, mask], dim=-1))

        if valid_frame is not None:
            gate = valid_frame.reshape(-1, 1).to(token.dtype)
            token = gate * token + (1.0 - gate) * self.null_token.unsqueeze(0)

        return self.norm(token).unsqueeze(1)


class SafetyPredictionHead(nn.Module):
    """Thesis modification 3: regress the six metrics from the safety token.

    "I trained a 2-layer MLP to predict 6 safety metrics from the final safety
    query embedding."

    Output is in unbounded ratio space so that the asymmetric Huber term can
    see values above 1.0.  Softplus keeps predictions non-negative, which every
    ratio is by construction (they are built from magnitudes), without capping
    them from above the way a sigmoid would.
    """

    def __init__(
        self, dim: int, num_metrics: int = NUM_SAFETY_METRICS, hidden: Optional[int] = None
    ) -> None:
        super().__init__()
        hidden = hidden or dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, num_metrics),
        )

    def forward(self, safety_token: torch.Tensor) -> torch.Tensor:
        """``[B, 1, D]`` or ``[B, D]`` -> ``[B, 6]`` predicted ratios."""
        if safety_token.dim() == 3:
            safety_token = safety_token.squeeze(1)
        # +1.0 bias inside softplus so the head starts near ratio 1.0 (at
        # threshold) rather than near 0, which is where most training targets
        # actually sit and avoids a long warm-up climbing off the floor.
        return nn.functional.softplus(self.mlp(safety_token) + 1.0)


class SafetyTemporalMemory(nn.Module):
    """Thesis modification 5: safety temporal cross-attention.

        H_ego(l+1) += cross_attention(
            Q = H_ego(l),
            K = [H_safety(t-N), ..., H_safety(t-1)],
            V = [H_safety(t-N), ..., H_safety(t-1)])

    "augmenting the FIFO queue of the temporal cross-attention module that
    stores the past task token states with the corresponding safety query
    embedding."

    The literal formula breaks at the start of an episode, when the queue is
    empty: attention over zero keys produces NaN (softmax of an empty set).
    Padding with zeros instead is not neutral either -- zero keys still receive
    attention mass and inject an arbitrary zero-vector value into the ego
    query.

    Fixed with an explicit key-padding mask plus one learnable "no history"
    key/value pair that is always present.  Early in an episode attention falls
    back onto that token; as real history accumulates it competes with the real
    keys on its merits.  See DOCUMENTATION.md, deviation D7.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.no_history = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.no_history, std=0.02)
        # Zero-initialised output gate: the module starts as an exact no-op, so
        # adding it cannot destabilise a converged DriveTransformer at the
        # beginning of fine-tuning.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        ego_query: torch.Tensor,
        safety_history: Optional[torch.Tensor],
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        ego_query
            ``[B, Q, D]`` planning/ego queries.
        safety_history
            ``[B, N, D]`` past safety embeddings, oldest first. May be None.
        history_mask
            ``[B, N]`` bool, True where the slot is *padding* (torch convention).
        """
        b = ego_query.shape[0]
        null_kv = self.no_history.expand(b, -1, -1)

        if safety_history is None or safety_history.shape[1] == 0:
            keys = null_kv
            key_padding_mask = None
        else:
            keys = torch.cat([null_kv, safety_history], dim=1)
            if history_mask is not None:
                null_mask = torch.zeros(
                    b, 1, dtype=torch.bool, device=history_mask.device
                )
                key_padding_mask = torch.cat([null_mask, history_mask], dim=1)
            else:
                key_padding_mask = None

        keys = self.norm_kv(keys)
        attended, _ = self.attn(
            self.norm_q(ego_query),
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return ego_query + self.gate * attended


class SafetyFIFO:
    """The FIFO queue of past safety embeddings, one per rollout.

    DriveTransformer keeps a FIFO of past scene query embeddings for temporal
    cross-attention; the thesis augments it with the safety embedding.  During
    training the history comes batched from the dataloader, so this class is
    only used for closed-loop inference, where frames arrive one at a time.
    """

    def __init__(self, length: int, dim: int, device: torch.device | str = "cpu") -> None:
        self.length = length
        self.dim = dim
        self.device = torch.device(device)
        self._buffer: list[torch.Tensor] = []

    def reset(self) -> None:
        self._buffer.clear()

    def push(self, embedding: torch.Tensor) -> None:
        """``embedding``: ``[D]`` or ``[1, D]`` or ``[1, 1, D]``."""
        self._buffer.append(embedding.detach().reshape(-1).to(self.device))
        if len(self._buffer) > self.length:
            self._buffer.pop(0)

    def get(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns ``([1, N, D], [1, N] padding mask)``, or ``(None, None)`` if empty."""
        if not self._buffer:
            return None, None
        stacked = torch.stack(self._buffer, dim=0).unsqueeze(0)
        mask = torch.zeros(
            1, stacked.shape[1], dtype=torch.bool, device=stacked.device
        )
        return stacked, mask

    def __len__(self) -> int:
        return len(self._buffer)
