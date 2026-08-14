"""Tests for the parts that do not need a CARLA server.

Run with:  pytest -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dav.losses.criterion import DAVCriterion
from dav.losses.safety_huber import SafetyHuberLoss, huber, thesis_literal_huber
from dav.metrics.evaluator import (
    EgoObservation,
    EpisodeCompliance,
    evaluate_frame,
    safety_vector,
    time_gap,
)
from dav.metrics.thresholds import (
    G,
    LON_ACCEL_MAX_COMFORTABLE,
    SAFETY_VECTOR_ORDER,
    WEATHER_PRESETS,
    DrivingContext,
    RoadClass,
    following_gap_for,
    is_heavy_vehicle,
    speed_limit_for,
)
from dav.models.drive_transformer import DAVConfig, build_model
from dav.models.heads import SafetyAwarePlanningHead
from dav.models.safety import SafetyFIFO


# ---------------------------------------------------------------------------
# Table 1 thresholds
# ---------------------------------------------------------------------------


def test_speed_factors_match_table_1():
    limit = 100 / 3.6  # 100 km/h
    assert speed_limit_for(DrivingContext(limit)) == pytest.approx(0.95 * limit)
    assert speed_limit_for(DrivingContext(limit, wet=True)) == pytest.approx(0.80 * limit)
    assert speed_limit_for(DrivingContext(limit, low_visibility=True)) == pytest.approx(0.70 * limit)
    assert speed_limit_for(DrivingContext(limit, night=True)) == pytest.approx(0.90 * limit)
    # The combined row supersedes the individual ones when all three hold.
    combined = DrivingContext(limit, wet=True, low_visibility=True, night=True)
    assert speed_limit_for(combined) == pytest.approx(0.75 * limit)


def test_most_restrictive_factor_wins_for_partial_conditions():
    limit = 100 / 3.6
    ctx = DrivingContext(limit, wet=True, night=True)  # not the combined row
    assert speed_limit_for(ctx) == pytest.approx(0.80 * limit)


def test_following_gap_multipliers():
    city = DrivingContext(30 / 3.6)
    assert following_gap_for(city) == pytest.approx(3.0)
    assert city.resolved_road_class() is RoadClass.CITY

    urban = DrivingContext(60 / 3.6)
    assert following_gap_for(urban) == pytest.approx(4.0)

    highway = DrivingContext(100 / 3.6)
    assert following_gap_for(highway) == pytest.approx(5.0)
    # x1.5 adverse weather and x1.3 heavy vehicle compose.
    both = DrivingContext(100 / 3.6, wet=True, lead_is_heavy_vehicle=True)
    assert following_gap_for(both) == pytest.approx(5.0 * 1.5 * 1.3)


def test_acceleration_thresholds_are_the_quoted_g_values():
    assert LON_ACCEL_MAX_COMFORTABLE == pytest.approx(1.47, abs=0.01)
    assert 0.30 * G == pytest.approx(2.94, abs=0.01)
    assert 0.47 * G == pytest.approx(4.61, abs=0.01)


def test_eleven_weather_presets():
    assert len(WEATHER_PRESETS) == 11
    assert len({w.carla_name for w in WEATHER_PRESETS}) == 11


def test_heavy_vehicle_detection():
    assert is_heavy_vehicle("vehicle.carlamotors.firetruck")
    assert is_heavy_vehicle("vehicle.mercedes.sprinter")
    assert not is_heavy_vehicle("vehicle.tesla.model3")


# ---------------------------------------------------------------------------
# Compliance evaluation
# ---------------------------------------------------------------------------


def _clean_observation(**kwargs) -> EgoObservation:
    base = dict(
        speed=8.0, lon_accel=0.5, lat_accel=0.5,
        lon_jerk=0.1, lat_jerk=0.1, lane_offset=0.05,
    )
    base.update(kwargs)
    return EgoObservation(**base)


def test_compliant_frame_has_no_violations():
    report = evaluate_frame(_clean_observation(), DrivingContext(13.9))
    assert report.compliant
    assert not report.hard_violation


def test_ratio_is_one_at_threshold_and_above_when_violating():
    ctx = DrivingContext(13.9)
    at = _clean_observation(lon_accel=LON_ACCEL_MAX_COMFORTABLE)
    ratios, _ = safety_vector(at, ctx)
    index = SAFETY_VECTOR_ORDER.index("lon_accel")
    assert ratios[index] == pytest.approx(1.0)
    # Exactly at threshold is still compliant; Table 1's rows read "<=".
    assert "lon_accel" not in evaluate_frame(at, ctx).violations

    over = _clean_observation(lon_accel=LON_ACCEL_MAX_COMFORTABLE * 1.2)
    assert "lon_accel" in evaluate_frame(over, ctx).violations


def test_following_gap_ratio_inverts_direction():
    """A gap shorter than required must land above 1.0, like every other metric."""
    ctx = DrivingContext(30 / 3.6)  # city, requires 3.0 s
    index = SAFETY_VECTOR_ORDER.index("following_gap")

    # 10 m at 10 m/s == 1.0 s gap, a third of what is required -> ratio 3.0.
    close = _clean_observation(speed=10.0, lead_distance=10.0)
    ratios, mask = safety_vector(close, ctx)
    assert mask[index] == 1.0
    assert ratios[index] == pytest.approx(3.0, rel=1e-3)

    # 60 m at 10 m/s == 6.0 s, twice what is required -> ratio 0.5.
    far = _clean_observation(speed=10.0, lead_distance=60.0)
    ratios, _ = safety_vector(far, ctx)
    assert ratios[index] == pytest.approx(0.5, rel=1e-3)


def test_following_gap_masked_when_no_lead_vehicle():
    ctx = DrivingContext(13.9)
    ratios, mask = safety_vector(_clean_observation(lead_distance=None), ctx)
    index = SAFETY_VECTOR_ORDER.index("following_gap")
    assert mask[index] == 0.0
    assert np.isfinite(ratios).all()


def test_time_gap_saturates_rather_than_diverging():
    stopped = _clean_observation(speed=0.0, lead_distance=30.0)
    assert time_gap(stopped) == pytest.approx(10.0)
    assert time_gap(_clean_observation(lead_distance=None)) is None


def test_hard_violation_flags_emergency_braking():
    ctx = DrivingContext(13.9)
    report = evaluate_frame(_clean_observation(lon_accel=-5.0), ctx)
    assert report.hard_violation
    assert "emergency_brake_exceeded" in report.violations


def test_red_light_rule_applies_only_at_the_stop_line():
    ctx = DrivingContext(13.9)
    approaching = _clean_observation(
        speed=8.0, at_red_light_or_stop=True, distance_to_stop_line=40.0
    )
    assert "red_light_stop" not in evaluate_frame(approaching, ctx).traffic_rules

    at_line = _clean_observation(
        speed=8.0, at_red_light_or_stop=True, distance_to_stop_line=0.5
    )
    rules = evaluate_frame(at_line, ctx).traffic_rules
    assert rules["red_light_stop"] is False


def test_episode_compliance_is_bounded_and_hard_gated():
    ctx = DrivingContext(13.9)
    agg = EpisodeCompliance()
    for _ in range(9):
        agg.add(evaluate_frame(_clean_observation(), ctx))
    agg.add(evaluate_frame(_clean_observation(lon_accel=-5.0), ctx))

    summary = agg.summary()
    assert 0.0 <= summary["compliance_score"] <= 1.0
    assert summary["compliance_score"] == pytest.approx(0.9)
    # Chapter 3: a single infraction fails the episode.
    assert agg.passed is False


# ---------------------------------------------------------------------------
# Safety loss
# ---------------------------------------------------------------------------


def test_standard_huber_is_continuous_at_delta():
    delta = 0.1
    below = huber(torch.tensor([delta - 1e-6]), delta)
    above = huber(torch.tensor([delta + 1e-6]), delta)
    assert torch.allclose(below, above, atol=1e-6)


def test_thesis_literal_formula_is_discontinuous():
    """Documents defect 3: the printed formula jumps 5x at |e| = delta."""
    delta, alpha = 0.1, 5.0
    target = torch.tensor([0.0])
    below = thesis_literal_huber(torch.tensor([delta - 1e-6]), target, delta, alpha)
    above = thesis_literal_huber(torch.tensor([delta + 1e-6]), target, delta, alpha)
    assert not torch.allclose(below, above, atol=1e-3)
    assert (above / below).item() == pytest.approx(alpha, rel=0.01)


def test_safety_loss_penalises_the_unsafe_side_harder():
    loss_fn = SafetyHuberLoss(delta=0.1, alpha=5.0)
    target = torch.full((1, 6), 0.9)   # just inside threshold
    mask = torch.ones(1, 6)

    over, _ = loss_fn(torch.full((1, 6), 1.4), target, mask)   # 0.5 above, unsafe
    under, _ = loss_fn(torch.full((1, 6), 0.4), target, mask)  # 0.5 below, safe
    assert over > under


def test_safety_loss_is_symmetric_while_both_sides_stay_below_threshold():
    """Inside the envelope the asymmetric term is inactive by construction."""
    loss_fn = SafetyHuberLoss(delta=0.1, alpha=5.0)
    target = torch.full((1, 6), 0.5)
    mask = torch.ones(1, 6)
    high, _ = loss_fn(torch.full((1, 6), 0.7), target, mask)
    low, _ = loss_fn(torch.full((1, 6), 0.3), target, mask)
    assert high.item() == pytest.approx(low.item(), rel=1e-5)


def test_masked_metrics_do_not_contribute():
    loss_fn = SafetyHuberLoss()
    target = torch.zeros(1, 6)
    pred = torch.zeros(1, 6)
    pred[0, 4] = 99.0  # a wild value on a masked-out metric

    mask = torch.ones(1, 6)
    mask[0, 4] = 0.0
    masked, _ = loss_fn(pred, target, mask)
    assert masked.item() == pytest.approx(0.0, abs=1e-6)


def test_safety_loss_gradient_flows():
    loss_fn = SafetyHuberLoss()
    pred = torch.full((2, 6), 1.5, requires_grad=True)
    loss, _ = loss_fn(pred, torch.full((2, 6), 0.5), torch.ones(2, 6))
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _tiny_config(**kwargs) -> DAVConfig:
    base = dict(
        backbone="resnet18", pretrained=False, num_cameras=6,
        image_size=(64, 128), dim=64, num_blocks=1, num_heads=4,
        num_agent_queries=8, num_map_queries=4, temporal_length=3,
    )
    base.update(kwargs)
    return DAVConfig(**base)


def _forward(model, cfg, batch=2, with_history=True):
    h, w = cfg.image_size
    kwargs = dict(
        images=torch.randn(batch, cfg.num_cameras, 3, h, w),
        ego_state=torch.randn(batch, 6),
        command=torch.randint(0, 6, (batch,)),
        safety_ratios=torch.rand(batch, 6) * 2,
        safety_mask=torch.ones(batch, 6),
        safety_valid=torch.ones(batch),
    )
    if with_history:
        kwargs.update(
            safety_history_ratios=torch.rand(batch, cfg.temporal_length, 6) * 2,
            safety_history_masks=torch.ones(batch, cfg.temporal_length, 6),
            safety_history_mask=torch.zeros(batch, cfg.temporal_length, dtype=torch.bool),
        )
    return model(**kwargs)


def test_forward_shapes():
    cfg = _tiny_config()
    out = _forward(build_model(cfg), cfg)
    assert out["trajectory"].shape == (2, cfg.horizon, 2)
    assert out["pred_safety_ratios"].shape == (2, 6)
    assert out["mode_trajectories"].shape == (2, 6, cfg.horizon, 2)
    assert out["safety_query"].shape == (2, 1, cfg.dim)


def test_predicted_ratios_are_non_negative():
    cfg = _tiny_config()
    out = _forward(build_model(cfg), cfg)
    assert (out["pred_safety_ratios"] >= 0).all()


def test_runs_with_empty_safety_history():
    """First frame of an episode: the queue is empty and must not produce NaN."""
    cfg = _tiny_config()
    out = _forward(build_model(cfg), cfg, with_history=False)
    assert torch.isfinite(out["trajectory"]).all()


def test_fully_padded_history_does_not_nan():
    """Every history slot masked out -- the null key must carry the attention."""
    cfg = _tiny_config()
    model = build_model(cfg)
    out = model(
        images=torch.randn(2, 6, 3, *cfg.image_size),
        ego_state=torch.randn(2, 6),
        command=torch.zeros(2, dtype=torch.long),
        safety_ratios=torch.rand(2, 6),
        safety_mask=torch.ones(2, 6),
        safety_valid=torch.zeros(2),
        safety_history_ratios=torch.zeros(2, cfg.temporal_length, 6),
        safety_history_masks=torch.zeros(2, cfg.temporal_length, 6),
        safety_history_mask=torch.ones(2, cfg.temporal_length, dtype=torch.bool),
    )
    assert torch.isfinite(out["trajectory"]).all()
    assert torch.isfinite(out["pred_safety_ratios"]).all()


def test_safety_temporal_module_starts_as_a_no_op():
    """The zero-init gate means adding modification 5 cannot perturb a warm model."""
    cfg = _tiny_config()
    model = build_model(cfg)
    ego = torch.randn(2, 1, cfg.dim)
    history = torch.randn(2, cfg.temporal_length, cfg.dim)
    out = model.safety_temporal(ego, history, None)
    assert torch.allclose(out, ego)


def test_ablation_flags_change_behaviour_not_shape():
    for flag in ("use_safety_pe", "use_safety_planning_head", "use_safety_temporal"):
        cfg = _tiny_config(**{flag: False})
        out = _forward(build_model(cfg), cfg)
        assert out["trajectory"].shape == (2, cfg.horizon, 2)


def test_safety_query_disabled_removes_the_head():
    cfg = _tiny_config(use_safety_query=False)
    out = _forward(build_model(cfg), cfg)
    assert "pred_safety_ratios" not in out
    assert out["trajectory"].shape == (2, cfg.horizon, 2)


def test_safety_gate_prefers_a_compliant_mode():
    """A high-scoring but threshold-violating mode must lose to a compliant one."""
    b, m, t = 1, 6, 4
    outputs = {
        "mode_trajectories": torch.arange(b * m * t * 2, dtype=torch.float32).reshape(b, m, t, 2),
        "mode_scores": torch.zeros(b, m),
        "mode_safety_ratios": torch.full((b, m, 6), 0.5),
    }
    outputs["mode_scores"][0, 3] = 5.0          # loudest mode ...
    outputs["mode_safety_ratios"][0, 3] = 2.0   # ... but unsafe
    outputs["mode_scores"][0, 1] = 1.0          # quieter and compliant

    gated = SafetyAwarePlanningHead.select(outputs, safety_gate=True)
    assert torch.allclose(gated, outputs["mode_trajectories"][0, 1].unsqueeze(0))

    ungated = SafetyAwarePlanningHead.select(outputs, safety_gate=False)
    assert torch.allclose(ungated, outputs["mode_trajectories"][0, 3].unsqueeze(0))


def test_backward_pass_reaches_every_safety_module():
    cfg = _tiny_config()
    model = build_model(cfg)
    out = _forward(model, cfg)
    (out["trajectory"].sum() + out["pred_safety_ratios"].sum()).backward()

    for name in ("safety_encoder", "safety_head", "safety_pe"):
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"{name} received no gradient"
        assert any(g.abs().sum() > 0 for g in grads), f"{name} gradient is all zero"


# ---------------------------------------------------------------------------
# FIFO
# ---------------------------------------------------------------------------


def test_fifo_evicts_oldest_and_reports_padding():
    fifo = SafetyFIFO(length=3, dim=8)
    assert fifo.get() == (None, None)

    for i in range(5):
        fifo.push(torch.full((8,), float(i)))
    assert len(fifo) == 3

    stacked, mask = fifo.get()
    assert stacked.shape == (1, 3, 8)
    assert not mask.any()
    # Oldest first: the two earliest pushes were evicted.
    assert stacked[0, 0, 0].item() == pytest.approx(2.0)
    assert stacked[0, -1, 0].item() == pytest.approx(4.0)

    fifo.reset()
    assert len(fifo) == 0


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


def test_criterion_produces_finite_loss_and_logs():
    cfg = _tiny_config()
    model = build_model(cfg)
    out = _forward(model, cfg)

    b = 2
    batch = {
        "trajectory_gt": torch.randn(b, cfg.horizon, 2),
        "agent_labels_gt": torch.randint(-1, 5, (b, 12)),
        "agent_boxes_gt": torch.randn(b, 12, 8),
        "map_labels_gt": torch.randint(-1, 3, (b, 6)),
        "map_points_gt": torch.randn(b, 6, cfg.map_points, 2),
        "safety_target": torch.rand(b, 6) * 2,
        "safety_target_mask": torch.ones(b, 6),
    }
    loss, logs = DAVCriterion(cfg.num_agent_classes, cfg.num_map_classes)(out, batch)
    assert torch.isfinite(loss)
    assert logs["loss/total"] == pytest.approx(float(loss), rel=1e-5)
    for key in ("loss/plan", "loss/safety", "metric/ade", "metric/fde"):
        assert math.isfinite(logs[key])


def test_criterion_handles_a_frame_with_no_ground_truth_objects():
    """All-padding targets must not crash the Hungarian matcher."""
    cfg = _tiny_config()
    out = _forward(build_model(cfg), cfg)
    b = 2
    batch = {
        "trajectory_gt": torch.randn(b, cfg.horizon, 2),
        "agent_labels_gt": torch.full((b, 12), -1),
        "agent_boxes_gt": torch.zeros(b, 12, 8),
        "map_labels_gt": torch.full((b, 6), -1),
        "map_points_gt": torch.zeros(b, 6, cfg.map_points, 2),
        "safety_target": torch.rand(b, 6),
        "safety_target_mask": torch.ones(b, 6),
    }
    loss, _ = DAVCriterion(cfg.num_agent_classes, cfg.num_map_classes)(out, batch)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_roundtrip(tmp_path):
    """Generate a synthetic dataset and read it back through DAVDataset."""
    import subprocess
    import sys
    from pathlib import Path

    from dav.data.dataset import DAVDataset

    root = tmp_path / "synth"
    script = Path(__file__).resolve().parent.parent / "scripts" / "make_synthetic_dataset.py"
    subprocess.run(
        [sys.executable, str(script), "--out", str(root), "--episodes", "2",
         "--frames", "40", "--width", "64", "--height", "36"],
        check=True, capture_output=True,
    )

    dataset = DAVDataset(root, horizon=3, history=2, image_size=(36, 64),
                         waypoint_stride=5)
    assert len(dataset) > 0

    item = dataset[0]
    assert item["images"].shape == (6, 3, 36, 64)
    assert item["trajectory_gt"].shape == (3, 2)
    assert item["safety_history_ratios"].shape == (2, 6)
    # Frame 0 of an episode has no previous measurement.
    assert item["safety_valid"].item() == 0.0
    assert item["safety_history_padding"].all()

    later = dataset[10]
    assert later["safety_valid"].item() == 1.0
    assert not later["safety_history_padding"].all()

    train, val = DAVDataset.split(root, val_fraction=0.5)
    assert set(train).isdisjoint(val)
    assert len(train) + len(val) == 2


# ---------------------------------------------------------------------------
# Collector (no simulator required)
# ---------------------------------------------------------------------------


def test_map_name_normalisation_avoids_needless_reloads():
    """The server boots into Town10HD_Opt; a Town10HD episode must not reload."""
    from dav.data.collector import _basename_map

    assert _basename_map("Carla/Maps/Town10HD_Opt") == "Town10HD"
    assert _basename_map("Carla/Maps/Town01") == "Town01"
    assert _basename_map("Town03_Opt") == "Town03"
    # A town whose real name merely contains "Opt" must survive intact.
    assert _basename_map("Carla/Maps/Optimus") == "Optimus"


def test_pedal_slew_is_derived_from_the_jerk_row_not_chosen():
    """The pedal is the torque command, so its slew rate *is* the jerk (D23)."""
    from dav.data import expert as ex
    from dav.metrics.thresholds import JERK_LON_ACCEPTABLE

    # The pedal must be able to follow a jerk-limited reference...
    follow_rate = 0.85 * JERK_LON_ACCEPTABLE / ex.PEDAL_ACCEL_GAIN
    assert ex.PEDAL_MAX_RATE >= follow_rate, (
        "pedal cannot even track the reference acceleration; the loop has no "
        "authority to slow down (this caused a collision with a stopped car)"
    )
    # ...with headroom for error correction, but not unlimited: the first
    # version used 1.2/s, implying 4.6 m/s^3 against a 0.60 row.
    assert ex.PEDAL_MAX_RATE <= 5.0 * follow_rate

    # The emergency path must actually be reachable: the commanded-acceleration
    # jerk budget has to let the demand reach the emergency trigger quickly.
    from dav.metrics.thresholds import BRAKE_DEFENSIVE_TARGET

    seconds_to_arm = BRAKE_DEFENSIVE_TARGET / ex.JERK_EMERGENCY
    assert seconds_to_arm < 0.5, (
        f"takes {seconds_to_arm:.1f}s for the commanded acceleration to reach "
        "the emergency threshold; comfort limiting must not gate braking"
    )


def test_crosstrack_correction_is_negative_feedback():
    """D29: CARLA is left-handed, so the crosstrack term must be subtracted.

    Reproduces the geometry without a simulator: place the vehicle to the right
    of a lane running along +x and check the commanded steer turns it left.
    Adding the term instead of subtracting it steers further right, which is
    the positive feedback that left a standing 0.208 m offset.
    """
    import math

    from dav.data.expert import DefensiveExpert

    lane_yaw = 0.0  # lane runs along +x
    # Vehicle displaced to the lane's right-hand side. In CARLA's left-handed
    # frame the right-hand normal of the lane is (-sin, cos) == (0, +1).
    dx, dy = 0.0, +1.5
    crosstrack = -dx * math.sin(lane_yaw) + dy * math.cos(lane_yaw)
    assert crosstrack > 0, "positive crosstrack must mean 'right of centre'"

    # Heading already aligned with the lane: the only correction is lateral.
    steer = DefensiveExpert.steer_law(
        heading_error=0.0, crosstrack=crosstrack, speed=5.0,
        crosstrack_integral=0.0,
        k_heading=0.8, k_crosstrack=0.35, k_crosstrack_integral=0.025,
    )
    assert steer < 0, "a car right of centre must be steered left, not right"

    # Mirror image steers the other way, and the law is odd in crosstrack.
    mirrored = DefensiveExpert.steer_law(
        heading_error=0.0, crosstrack=-crosstrack, speed=5.0,
        crosstrack_integral=0.0,
        k_heading=0.8, k_crosstrack=0.35, k_crosstrack_integral=0.025,
    )
    assert mirrored == pytest.approx(-steer)

    # The integral pushes the same way as the error it accumulated.
    wound = DefensiveExpert.steer_law(
        heading_error=0.0, crosstrack=crosstrack, speed=5.0,
        crosstrack_integral=1.0,
        k_heading=0.8, k_crosstrack=0.35, k_crosstrack_integral=0.025,
    )
    assert wound < steer, "a positive accumulated offset must add left steer"


def test_running_a_red_light_is_scored_as_a_violation():
    """D64: crossing a stop line on red must register, not just idling near it.

    The old rule asked only "within 1 m of the line and moving?", so driving
    *through* a red light registered the two or three frames spent inside that
    band -- 0.23% of an episode for a manoeuvre a human would call running a
    red light. A per-frame proximity test cannot see a crossing at all.
    """
    from dav.metrics.evaluator import EgoObservation, evaluate_frame
    from dav.metrics.thresholds import DrivingContext

    ctx = DrivingContext(speed_limit_ms=13.9)
    cruising = dict(
        lon_accel=0.0, lat_accel=0.0, lon_jerk=0.0, lat_jerk=0.0, lane_offset=0.0
    )

    # Driving through the line at speed, well past it: the proximity band no
    # longer applies, so only the crossing event can catch this.
    ran = evaluate_frame(
        EgoObservation(
            speed=8.0, at_red_light_or_stop=True, distance_to_stop_line=-6.0,
            ran_red_light=True, **cruising,
        ),
        ctx,
    )
    assert "traffic:red_light_run" in ran.violations

    # Stopped at the line is compliant.
    stopped = evaluate_frame(
        EgoObservation(
            speed=0.0, at_red_light_or_stop=True, distance_to_stop_line=1.0,
            **cruising,
        ),
        ctx,
    )
    assert "traffic:red_light_stop" not in stopped.violations
    assert "traffic:red_light_run" not in stopped.violations

    # Rolling right at the line without stopping violates the yield rule.
    rolling = evaluate_frame(
        EgoObservation(
            speed=3.0, at_red_light_or_stop=True, distance_to_stop_line=1.0,
            **cruising,
        ),
        ctx,
    )
    assert "traffic:red_light_stop" in rolling.violations

    # Still decelerating on the approach is NOT a violation: the band is
    # deliberately narrow so that braking to a halt -- which requires moving
    # through the approach -- is not scored as failing to stop (D66).
    braking = evaluate_frame(
        EgoObservation(
            speed=4.0, at_red_light_or_stop=True, distance_to_stop_line=6.0,
            **cruising,
        ),
        ctx,
    )
    assert "traffic:red_light_stop" not in braking.violations

    # Still approaching from 40 m back is not yet required to be stationary.
    approaching = evaluate_frame(
        EgoObservation(
            speed=8.0, at_red_light_or_stop=True, distance_to_stop_line=40.0,
            **cruising,
        ),
        ctx,
    )
    assert "traffic:red_light_stop" not in approaching.violations


def test_nan_actor_state_cannot_poison_the_yield_rule():
    """D44: a NaN answers 'no' to every comparison, so both branches fail.

    CARLA returns a non-finite transform for an actor destroyed mid-tick. The
    yield rule is "clear of pedestrians OR stopped"; with a NaN distance both
    disjuncts evaluate False and every subsequent frame is a violation. This
    cost 142 consecutive frames -- 35.5% of an episode -- and was invisible
    because it looked exactly like the expert driving badly.
    """
    import math

    from dav.data.expert import _is_finite_location
    from dav.metrics.evaluator import EgoObservation, evaluate_frame
    from dav.metrics.thresholds import DrivingContext

    class _Loc:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    assert _is_finite_location(_Loc(1.0, 2.0, 3.0))
    assert not _is_finite_location(_Loc(float("nan"), 0.0, 0.0))
    assert not _is_finite_location(_Loc(0.0, float("inf"), 0.0))

    # Demonstrate the failure the guard prevents: NaN defeats both branches.
    nan = float("nan")
    assert not (nan >= 5.0)
    assert not (nan < 0.14)

    # With the pedestrian filtered out, a moving ego on a clear road is clean.
    report = evaluate_frame(
        EgoObservation(
            speed=5.0, lon_accel=0.0, lat_accel=0.0, lon_jerk=0.0, lat_jerk=0.0,
            lane_offset=0.0, nearest_pedestrian_distance=None,
        ),
        DrivingContext(speed_limit_ms=13.9),
    )
    assert "traffic:pedestrian_yield" not in report.violations


def test_closest_point_of_approach_detects_a_crossing_conflict():
    """D30: the give-way check must be lane-agnostic.

    Reproduces the geometry that caused three consecutive junction collisions:
    ego heading +x at 6.4 m/s, another vehicle crossing on +y, on a course that
    puts them in the same place at the same time. A same-lane test cannot see
    this; the closest-point-of-approach test must.
    """
    from dav.data.expert import (
        CONFLICT_CLEARANCE, CONFLICT_HORIZON, DefensiveExpert,
    )

    cpa = DefensiveExpert.closest_point_of_approach

    # Ego at the origin heading +x at 6.4 m/s; crosser ahead-right heading +y,
    # on a course that puts both at (25.6, 0) in 4 s.
    ego_vx, ego_vy = 6.4, 0.0
    hit = cpa(25.6, -24.0, 0.0 - ego_vx, 6.0 - ego_vy)
    assert hit is not None
    t, miss = hit
    assert t == pytest.approx(4.0), f"expected conflict in 4 s, got {t:.2f}"
    assert miss < CONFLICT_CLEARANCE and t < CONFLICT_HORIZON
    assert miss == pytest.approx(0.0, abs=1e-6)

    # Traffic in the next lane over, travelling parallel at the same speed,
    # never closes: no relative motion at all.
    assert cpa(20.0, 3.5, 0.0, 0.0) is None

    # A vehicle that has already crossed and is departing must not trip it.
    assert cpa(-10.0, 0.0, -5.0, 0.0) is None

    # Same crossing geometry but passing well clear: detected, but outside the
    # clearance, so the caller lets it through.
    clear = cpa(25.6, -24.0, 0.0 - ego_vx, 3.0 - ego_vy)
    assert clear is not None and clear[1] > CONFLICT_CLEARANCE


def test_episode_schedule_groups_towns_without_losing_coverage():
    from dav.data.collector import CollectConfig, _episode_schedule
    from dav.metrics.thresholds import TOWNS, WEATHER_PRESETS

    cfg = CollectConfig(out_dir="/tmp/unused", episodes=50)
    schedule = _episode_schedule(cfg)

    assert len(schedule) == 50
    # Every town appears in one contiguous run -> one map load per town.
    towns = [town for town, _ in schedule]
    assert len(set(towns)) == len(TOWNS)
    runs = sum(1 for a, b in zip(towns, towns[1:]) if a != b) + 1
    assert runs == len(set(towns))
    # Weather coverage is not sacrificed to get that grouping.
    assert len({w.carla_name for _, w in schedule}) == len(WEATHER_PRESETS)
