"""The full training objective: DriveTransformer's tasks plus the safety term.

    L = w_plan * L_plan
      + w_agent * L_agent
      + w_map * L_map
      + w_safety * L_safety_huber        <- thesis modification 3
      + w_mode_safety * L_mode_safety    <- the per-mode safety supervision (D10)

Set-prediction tasks (agents, map elements) use Hungarian matching, as in the
DETR-derived detection heads DriveTransformer inherits: predictions are an
unordered set, so the loss has to find the best assignment before it can score
anything.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .safety_huber import SafetyHuberLoss


class HungarianMatcher(nn.Module):
    """Cheapest one-to-one assignment between predictions and ground truth."""

    def __init__(self, cost_class: float = 1.0, cost_box: float = 5.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_box = cost_box

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,  # [B, Q, C+1]
        pred_boxes: torch.Tensor,  # [B, Q, K]
        target_labels: torch.Tensor,  # [B, T]  (-1 == padding)
        target_boxes: torch.Tensor,  # [B, T, K]
    ):
        b, q = pred_logits.shape[:2]
        probability = pred_logits.softmax(dim=-1)
        indices = []

        for i in range(b):
            valid = target_labels[i] >= 0
            n = int(valid.sum())
            if n == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long),
                    )
                )
                continue

            labels = target_labels[i][valid]
            boxes = target_boxes[i][valid]

            cost = (
                -self.cost_class * probability[i][:, labels]
                + self.cost_box * torch.cdist(pred_boxes[i], boxes, p=1)
            )
            rows, cols = linear_sum_assignment(cost.detach().float().cpu().numpy())
            # Map back to indices in the *unpadded* target tensor.
            target_positions = torch.nonzero(valid, as_tuple=False).squeeze(1)
            indices.append(
                (torch.as_tensor(rows, dtype=torch.long),
                 target_positions[torch.as_tensor(cols, dtype=torch.long)])
            )
        return indices


class DAVCriterion(nn.Module):
    def __init__(
        self,
        num_agent_classes: int = 10,
        num_map_classes: int = 3,
        weights: Optional[Dict[str, float]] = None,
        safety_delta: float = 0.1,
        safety_alpha: float = 5.0,
        literal_thesis_huber: bool = False,
    ) -> None:
        super().__init__()
        self.weights = {
            "plan": 1.0,
            "agent_cls": 1.0,
            "agent_box": 1.0,
            "map_cls": 0.5,
            "map_pts": 0.5,
            "safety": 1.0,
            "mode_safety": 0.5,
            "mode_score": 0.2,
            **(weights or {}),
        }
        self.num_agent_classes = num_agent_classes
        self.num_map_classes = num_map_classes
        self.matcher = HungarianMatcher()
        self.safety_loss = SafetyHuberLoss(
            delta=safety_delta,
            alpha=safety_alpha,
            literal_thesis_formula=literal_thesis_huber,
        )

    # ------------------------------------------------------------------

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logs: Dict[str, float] = {}
        total = outputs["trajectory"].new_zeros(())

        # -- Planning ---------------------------------------------------
        # Supervise every mode, weighted towards the closest one
        # (winner-take-all): with a single best-mode loss the other five modes
        # never receive gradient and collapse to noise, and with a uniform loss
        # all six converge to the same average trajectory. The mixture keeps
        # the modes distinct while still training all of them.
        target = batch["trajectory_gt"]  # [B, T, 2]
        modes = outputs["mode_trajectories"]  # [B, M, T, 2]
        per_mode = F.smooth_l1_loss(
            modes, target.unsqueeze(1).expand_as(modes), reduction="none"
        ).mean(dim=(2, 3))  # [B, M]

        best = per_mode.argmin(dim=1)
        winner = per_mode.gather(1, best.unsqueeze(1)).squeeze(1).mean()
        spread = per_mode.mean()
        plan_loss = winner + 0.1 * spread
        total = total + self.weights["plan"] * plan_loss
        logs["loss/plan"] = float(plan_loss)
        logs["loss/plan_winner"] = float(winner)

        # The mode score is trained to identify which mode actually won.
        mode_score_loss = F.cross_entropy(outputs["mode_scores"], best)
        total = total + self.weights["mode_score"] * mode_score_loss
        logs["loss/mode_score"] = float(mode_score_loss)

        # Final selected trajectory, for reporting rather than optimisation.
        with torch.no_grad():
            l2 = torch.norm(outputs["trajectory"] - target, dim=-1)
            logs["metric/ade"] = float(l2.mean())
            logs["metric/fde"] = float(l2[:, -1].mean())

        # -- Agents -----------------------------------------------------
        agent_loss, agent_logs = self._set_loss(
            outputs["agent_logits"],
            outputs["agent_boxes"],
            batch["agent_labels_gt"],
            batch["agent_boxes_gt"],
            self.num_agent_classes,
            prefix="agent",
        )
        total = total + self.weights["agent_cls"] * agent_logs["cls"] + \
            self.weights["agent_box"] * agent_logs["box"]
        logs["loss/agent_cls"] = float(agent_logs["cls"])
        logs["loss/agent_box"] = float(agent_logs["box"])

        # -- Map --------------------------------------------------------
        map_points = outputs["map_points"]
        b, q, p, _ = map_points.shape
        map_loss, map_logs = self._set_loss(
            outputs["map_logits"],
            map_points.reshape(b, q, p * 2),
            batch["map_labels_gt"],
            batch["map_points_gt"].reshape(b, -1, p * 2),
            self.num_map_classes,
            prefix="map",
        )
        total = total + self.weights["map_cls"] * map_logs["cls"] + \
            self.weights["map_pts"] * map_logs["box"]
        logs["loss/map_cls"] = float(map_logs["cls"])
        logs["loss/map_pts"] = float(map_logs["box"])

        # -- Safety (thesis modification 3) -----------------------------
        if "pred_safety_ratios" in outputs:
            safety_loss, safety_logs = self.safety_loss(
                outputs["pred_safety_ratios"],
                batch["safety_target"],
                batch["safety_target_mask"],
            )
            total = total + self.weights["safety"] * safety_loss
            logs["loss/safety"] = float(safety_loss)
            logs.update({k: float(v) for k, v in safety_logs.items()})

            # Per-mode safety prediction (D10): each mode is asked what safety
            # state it would produce, supervised with the same target. This is
            # what gives mode selection a direct safety signal.
            if "mode_safety_ratios" in outputs:
                m = outputs["mode_safety_ratios"].shape[1]
                mode_safety_loss, _ = self.safety_loss(
                    outputs["mode_safety_ratios"],
                    batch["safety_target"].unsqueeze(1).expand(-1, m, -1),
                    batch["safety_target_mask"].unsqueeze(1).expand(-1, m, -1),
                )
                total = total + self.weights["mode_safety"] * mode_safety_loss
                logs["loss/mode_safety"] = float(mode_safety_loss)

        logs["loss/total"] = float(total)
        return total, logs

    # ------------------------------------------------------------------

    def _set_loss(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        num_classes: int,
        prefix: str,
    ):
        indices = self.matcher(pred_logits, pred_boxes, target_labels, target_boxes)

        # Everything unmatched is the "no object" class, index ``num_classes``.
        target_class = torch.full(
            pred_logits.shape[:2], num_classes, dtype=torch.long, device=pred_logits.device
        )
        box_losses = []
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            pred_idx = pred_idx.to(pred_logits.device)
            tgt_idx = tgt_idx.to(pred_logits.device)
            target_class[i, pred_idx] = target_labels[i][tgt_idx].clamp(0, num_classes - 1)
            box_losses.append(
                F.l1_loss(pred_boxes[i][pred_idx], target_boxes[i][tgt_idx], reduction="mean")
            )

        cls_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]), target_class.reshape(-1)
        )
        box_loss = (
            torch.stack(box_losses).mean()
            if box_losses
            else pred_boxes.new_zeros(())
        )
        return cls_loss + box_loss, {"cls": cls_loss, "box": box_loss}
