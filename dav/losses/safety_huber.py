"""The safety Huber term added to the DriveTransformer loss.

Chapter 3 of the thesis specifies:

    L_safety_huber = sum_i w_i * Hber(x_hat_i, x_i, delta=0.1, alpha=5.0)

                     0.5 (x_hat - x)^2,                for |x_hat - x|
    Hber(...)  =     alpha * delta * (|x_hat - x| - 0.5 delta),  for x_hat > x
                     delta * (|x_hat - x| - 0.5 delta),          otherwise

together with the prose "penalises values predicted above the metric threshold
by 5 times and does not penalise it if the values are below the threshold".

That specification cannot be implemented as written.  Four defects, and what
this module does instead (all four are restated in DOCUMENTATION.md as D1-D4):

1.  The first branch's guard is incomplete -- it reads "for |x_hat - x|" with
    no comparison.  Standard Huber intends ``<= delta``.

2.  The branches overlap.  A prediction with ``|x_hat - x| <= delta`` *and*
    ``x_hat > x`` satisfies branches 1 and 2 simultaneously, so the piecewise
    function is not well defined.

3.  The function is discontinuous.  Ordinary Huber is continuous at
    ``|e| = delta`` because ``0.5 delta^2 == delta * (delta - 0.5 delta)``.
    Multiplying only the linear branch by ``alpha = 5`` breaks that: at the
    join the quadratic branch gives ``0.5 delta^2`` and the linear branch gives
    ``2.5 delta^2``, a 5x jump in the loss and an unbounded jump in its
    gradient.  Training on it oscillates.

4.  Formula and prose disagree about the reference point.  The formula's
    alpha branch triggers on ``x_hat > x`` -- above the *ground-truth value* --
    while the prose says "above the metric *threshold*".  These are different
    conditions, and only the prose one expresses defensive driving.  Taking the
    prose literally, though, leaves the loss identically zero on every safe
    frame, so the auxiliary head would receive no gradient at all from the
    (large) majority of the data and would never learn to regress the metrics.

The implementation below keeps the intent -- asymmetric penalty, 5x on the
unsafe side, Huber-shaped, delta = 0.1, alpha = 5.0 -- and is well defined:

    L = sum_i w_i * m_i * [ huber_d(r_hat_i - r_i)
                            + alpha * huber_d(relu(r_hat_i - 1) - relu(r_i - 1)) ]

* Everything is in *threshold-normalised ratio space* (``r = value /
  threshold``, so r = 1 is exactly at threshold).  This is what makes a single
  ``delta = 0.1`` meaningful across all six metrics: in raw SI units the six
  span jerk at 0.6 m/s^3 and speed at ~30 m/s, so a fixed delta would put
  speed permanently in the linear regime and jerk permanently in the quadratic
  one.  Normalised, delta = 0.1 means "10% of threshold" for every metric.

* The first term is an ordinary symmetric Huber.  It keeps the auxiliary
  regression alive on safe frames (defect 4).

* The second term is the asymmetric part.  ``relu(r - 1)`` is the amount by
  which the metric exceeds its threshold, so this term is exactly zero while
  both prediction and target are within threshold, and grows at 5x weight
  above it.  This is the prose reading, and it is continuous everywhere with a
  single kink at r = 1 (the same kind of kink ReLU already has).

* ``m_i`` is the validity mask, which zeroes the following-distance metric on
  frames with no lead vehicle.

The literal formula is also provided, as ``thesis_literal_huber``, so the
defect can be reproduced and ablated rather than merely asserted.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics.thresholds import SAFETY_METRIC_SPECS, SAFETY_VECTOR_ORDER

DEFAULT_DELTA = 0.1
DEFAULT_ALPHA = 5.0


def huber(error: torch.Tensor, delta: float = DEFAULT_DELTA) -> torch.Tensor:
    """Standard, continuous Huber. Quadratic within delta, linear outside."""
    abs_e = error.abs()
    return torch.where(
        abs_e <= delta,
        0.5 * error * error,
        delta * (abs_e - 0.5 * delta),
    )


def thesis_literal_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = DEFAULT_DELTA,
    alpha: float = DEFAULT_ALPHA,
) -> torch.Tensor:
    """The formula exactly as printed in the thesis.

    Provided for ablation only -- it is discontinuous at ``|e| = delta`` (see
    defect 3 in the module docstring) and will destabilise training.  Branch
    order is resolved as written top-to-bottom, which is the only reading that
    makes the overlapping guards deterministic.
    """
    error = pred - target
    abs_e = error.abs()
    quadratic = 0.5 * error * error
    linear_penalised = alpha * delta * (abs_e - 0.5 * delta)
    linear_plain = delta * (abs_e - 0.5 * delta)
    return torch.where(
        abs_e <= delta,
        quadratic,
        torch.where(pred > target, linear_penalised, linear_plain),
    )


class SafetyHuberLoss(nn.Module):
    """Asymmetric threshold-aware Huber over the six safety metrics.

    Inputs and targets are violation ratios (see
    ``dav.metrics.evaluator.safety_vector``), shaped ``[B, 6]`` or ``[B, T, 6]``.
    """

    def __init__(
        self,
        delta: float = DEFAULT_DELTA,
        alpha: float = DEFAULT_ALPHA,
        weights: Optional[Dict[str, float]] = None,
        literal_thesis_formula: bool = False,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.literal = literal_thesis_formula

        w = weights or {n: SAFETY_METRIC_SPECS[n].weight for n in SAFETY_VECTOR_ORDER}
        weight_vec = torch.tensor(
            [w.get(n, 1.0) for n in SAFETY_VECTOR_ORDER], dtype=torch.float32
        )
        self.register_buffer("weight_vec", weight_vec)

    def forward(
        self,
        pred_ratios: torch.Tensor,
        target_ratios: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns (scalar loss, per-metric mean losses for logging)."""
        if mask is None:
            mask = torch.ones_like(target_ratios)

        if self.literal:
            per_metric = thesis_literal_huber(
                pred_ratios, target_ratios, self.delta, self.alpha
            )
        else:
            regression = huber(pred_ratios - target_ratios, self.delta)
            # Exceedance above threshold: zero while inside the envelope.
            exceed_pred = F.relu(pred_ratios - 1.0)
            exceed_target = F.relu(target_ratios - 1.0)
            asymmetric = huber(exceed_pred - exceed_target, self.delta)
            per_metric = regression + self.alpha * asymmetric

        per_metric = per_metric * mask * self.weight_vec

        # Normalise by the number of *valid* entries, not the tensor size, so
        # that batches with many lead-vehicle-free frames are not down-weighted.
        denom = (mask * self.weight_vec).sum().clamp_min(1e-6)
        loss = per_metric.sum() / denom

        with torch.no_grad():
            flat = per_metric.reshape(-1, per_metric.shape[-1])
            flat_mask = mask.reshape(-1, mask.shape[-1])
            counts = flat_mask.sum(dim=0).clamp_min(1e-6)
            means = flat.sum(dim=0) / counts
            logs = {
                f"safety_huber/{name}": means[i].detach()
                for i, name in enumerate(SAFETY_VECTOR_ORDER)
            }
        return loss, logs


class SafetyViolationMetrics(nn.Module):
    """Non-differentiable diagnostics for the GUI: how often is the model unsafe?"""

    @torch.no_grad()
    def forward(
        self,
        pred_ratios: torch.Tensor,
        target_ratios: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if mask is None:
            mask = torch.ones_like(target_ratios)
        out: Dict[str, float] = {}
        for i, name in enumerate(SAFETY_VECTOR_ORDER):
            m = mask[..., i]
            valid = m.sum().clamp_min(1e-6)
            pred_bad = ((pred_ratios[..., i] > 1.0).float() * m).sum() / valid
            true_bad = ((target_ratios[..., i] > 1.0).float() * m).sum() / valid
            out[f"violation_rate_pred/{name}"] = float(pred_bad)
            out[f"violation_rate_true/{name}"] = float(true_bad)
            err = ((pred_ratios[..., i] - target_ratios[..., i]).abs() * m).sum() / valid
            out[f"ratio_mae/{name}"] = float(err)
        return out
