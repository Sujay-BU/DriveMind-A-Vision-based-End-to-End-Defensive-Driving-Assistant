"""Per-frame defensive-driving compliance scoring.

Two things are produced from one ego-state observation:

``safety_vector``  the six-dimensional violation-ratio vector consumed by the
                   safety query and the safety Huber loss. 1.0 == exactly at
                   threshold, > 1.0 == unsafe.

``ComplianceReport`` the full Table 1 audit (all six metric categories,
                   including the traffic-rule row that the safety vector
                   deliberately excludes) used for evaluation and the GUI.

The thesis excludes the traffic-rule category from the safety token ("the
scalar metric data (except the traffic rules metric)") because those rows are
discrete events rather than continuous scalars. They are still scored here,
because Chapter 3 requires them for the hard-constraint check that gates the
dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .thresholds import (
    BRAKE_MAX_EMERGENCY,
    CROSSWALK_APPROACH_SPEED,
    FOLLOWING_GAP_CAP_S,
    JERK_HARD_LIMIT,
    LANE_DEPARTURE_MAX,
    LANE_OFFSET_MAX,
    LAT_ACCEL_HARSH,
    LON_ACCEL_HARSH,
    PEDESTRIAN_YIELD_DISTANCE,
    RED_LIGHT_STOP_BAND,
    SAFETY_VECTOR_ORDER,
    YELLOW_LIGHT_DECEL_DISTANCE,
    YIELD_SPEED_RED_LIGHT,
    DrivingContext,
    safety_thresholds_for,
)

#: Ratio strictly above this counts as a violation. Exactly 1.0 is "at
#: threshold", which Table 1 treats as still compliant (the rows read "<=").
VIOLATION_RATIO = 1.0


@dataclass
class EgoObservation:
    """One frame of ego state, in the ego (body) frame unless noted.

    All fields are directly measurable from CARLA, and all except
    ``lead_*`` are also directly measurable from a real vehicle's own IMU and
    speedometer -- which is what makes the safety query causally valid at
    inference time. See DOCUMENTATION.md, deviation D5.
    """

    speed: float  # m/s, scalar forward speed
    lon_accel: float  # m/s^2, signed (+ accelerating, - braking)
    lat_accel: float  # m/s^2, signed
    lon_jerk: float  # m/s^3, signed
    lat_jerk: float  # m/s^3, signed
    lane_offset: float  # m, absolute lateral deviation from lane centre

    #: Distance in metres to the lead vehicle, or None when the lane ahead is clear.
    lead_distance: Optional[float] = None
    lead_speed: Optional[float] = None  # m/s, lead vehicle's forward speed

    # Traffic-rule context. All optional; a None means "not applicable to this frame".
    at_red_light_or_stop: bool = False
    distance_to_stop_line: Optional[float] = None
    at_yellow_light: bool = False
    is_decelerating: bool = False
    nearest_pedestrian_distance: Optional[float] = None
    approaching_crosswalk: bool = False
    #: True on the frame the ego crosses a stop line while the light is red
    #: (D64). Detected by the expert, which has the previous frame's signed
    #: distance; a per-frame proximity test cannot see a crossing.
    ran_red_light: bool = False


@dataclass
class ComplianceReport:
    """Full Table 1 audit for a single frame."""

    #: Continuous violation ratios, keyed by metric name. > 1.0 == unsafe.
    ratios: Dict[str, float] = field(default_factory=dict)
    #: Thresholds actually applied, for display.
    thresholds: Dict[str, float] = field(default_factory=dict)
    #: Names of the metrics violated on this frame (continuous and discrete).
    violations: List[str] = field(default_factory=list)
    #: Discrete traffic-rule outcomes; True == compliant.
    traffic_rules: Dict[str, bool] = field(default_factory=dict)
    #: True when the frame breaches a limit Table 1 marks as never-exceed.
    hard_violation: bool = False

    @property
    def compliant(self) -> bool:
        return not self.violations

    def worst_ratio(self) -> float:
        return max(self.ratios.values(), default=0.0)


def _finite(x: float, default: float = 0.0) -> float:
    return x if math.isfinite(x) else default


def time_gap(observation: EgoObservation) -> Optional[float]:
    """Time gap in seconds to the lead vehicle, or None if there is no lead.

    Defined as bumper-to-bumper distance divided by ego speed, which is the
    "N second gap" convention the DMV manuals in refs [21-23] use.  At very low
    ego speed the quotient explodes, so it saturates at the cap; that is
    physically right (a stationary car has an unbounded time gap) and keeps the
    dataset free of infinities.
    """
    if observation.lead_distance is None:
        return None
    if observation.speed < 0.1:
        return FOLLOWING_GAP_CAP_S
    return min(observation.lead_distance / observation.speed, FOLLOWING_GAP_CAP_S)


def safety_vector(
    observation: EgoObservation, ctx: DrivingContext
) -> tuple[np.ndarray, np.ndarray]:
    """Build the six-dim violation-ratio vector and its validity mask.

    Returns
    -------
    ratios : float32[6]
        Violation ratios in ``SAFETY_VECTOR_ORDER``. 1.0 == at threshold.
    mask : float32[6]
        1.0 where the metric is defined on this frame, 0.0 where it is not.
        Only ``following_gap`` is ever masked out (no lead vehicle), but the
        mask is full width so the loss can index it without special cases.
    """
    thresholds = safety_thresholds_for(ctx)
    ratios = np.zeros(len(SAFETY_VECTOR_ORDER), dtype=np.float32)
    mask = np.ones(len(SAFETY_VECTOR_ORDER), dtype=np.float32)

    # Five "smaller is safer" metrics: |measured| / threshold.
    # Magnitudes are used because Table 1's acceleration and jerk rows bound
    # the magnitude, not the signed value -- braking at 3 m/s^2 and
    # accelerating at 3 m/s^2 are equally far outside the comfort envelope.
    measured = {
        "lon_accel": abs(observation.lon_accel),
        "lat_accel": abs(observation.lat_accel),
        "lon_jerk": abs(observation.lon_jerk),
        "lat_jerk": abs(observation.lat_jerk),
        "speed": max(observation.speed, 0.0),
    }
    for i, name in enumerate(SAFETY_VECTOR_ORDER):
        if name == "following_gap":
            continue
        thr = thresholds[name]
        ratios[i] = _finite(measured[name] / thr) if thr > 0 else 0.0

    # Following distance inverts: required / measured, so that a gap shorter
    # than required lands above 1.0 like every other metric.
    gap_index = SAFETY_VECTOR_ORDER.index("following_gap")
    gap = time_gap(observation)
    if gap is None:
        ratios[gap_index] = 0.0
        mask[gap_index] = 0.0
    else:
        required = thresholds["following_gap"]
        ratios[gap_index] = _finite(required / max(gap, 1e-3))

    return ratios, mask


def evaluate_frame(
    observation: EgoObservation, ctx: DrivingContext
) -> ComplianceReport:
    """Score one frame against every row of Table 1."""
    thresholds = safety_thresholds_for(ctx)
    ratios_vec, mask = safety_vector(observation, ctx)

    report = ComplianceReport()
    for i, name in enumerate(SAFETY_VECTOR_ORDER):
        if mask[i] == 0.0:
            continue
        report.ratios[name] = float(ratios_vec[i])
        report.thresholds[name] = float(thresholds[name])
        if ratios_vec[i] > VIOLATION_RATIO:
            report.violations.append(name)

    # D48: lane keeping is scored as a *departure*, not as a centring error.
    #
    # Table 1's six metrics are speed, following distance, longitudinal and
    # lateral acceleration, jerk, and traffic rules -- lane position is not
    # among them. The 0.30 m tolerance previously used here was introduced by
    # this implementation, filed under "traffic rules", and no traffic rule
    # requires a vehicle to stay within 30 cm of lane centre. It is a
    # ride-quality target, and using it to gate compliance was measuring
    # something the specification never asked for: it accounted for ~16% of all
    # violated frames.
    #
    # The rule a traffic code does impose is not leaving your lane. A CARLA
    # lane is 3.5 m and the vehicle is ~1.8 m wide, so the centre may deviate
    # about 0.85 m before a wheel crosses the marking. That is the threshold
    # that gates compliance now.
    #
    # The strict figure is still computed and reported so nothing is hidden --
    # it is simply diagnostic. Read ``lane_offset`` for quality of lane
    # centring and ``lane_departure`` for whether the vehicle left its lane.
    lane_ratio = abs(observation.lane_offset) / LANE_OFFSET_MAX
    report.ratios["lane_offset"] = float(lane_ratio)
    report.thresholds["lane_offset"] = LANE_OFFSET_MAX

    departure_ratio = abs(observation.lane_offset) / LANE_DEPARTURE_MAX
    report.ratios["lane_departure"] = float(departure_ratio)
    report.thresholds["lane_departure"] = LANE_DEPARTURE_MAX
    if departure_ratio > VIOLATION_RATIO:
        report.violations.append("lane_departure")

    # Never-exceed limits. Table 1 gives these as separate, stricter rows than
    # the comfort thresholds; breaching one is a hard failure rather than a
    # degree of discomfort.
    hard_checks = {
        "lon_accel_harsh": abs(observation.lon_accel) > LON_ACCEL_HARSH,
        "emergency_brake_exceeded": (
            observation.lon_accel < 0 and abs(observation.lon_accel) > BRAKE_MAX_EMERGENCY
        ),
        "lat_accel_harsh": abs(observation.lat_accel) > LAT_ACCEL_HARSH,
        "jerk_hard_limit": (
            max(abs(observation.lon_jerk), abs(observation.lat_jerk)) > JERK_HARD_LIMIT
        ),
    }
    for name, breached in hard_checks.items():
        if breached:
            report.violations.append(name)
            report.hard_violation = True

    report.traffic_rules = _evaluate_traffic_rules(observation)
    for name, ok in report.traffic_rules.items():
        if not ok:
            report.violations.append(f"traffic:{name}")
            report.hard_violation = True

    return report


def _evaluate_traffic_rules(obs: EgoObservation) -> Dict[str, bool]:
    """Row 6 of Table 1. Only applicable rules appear in the result."""
    rules: Dict[str, bool] = {}

    if obs.at_red_light_or_stop:
        # "Red light/stop sign yield: < 0.5 km/h". Enforced from the stop line
        # up to it -- a car still approaching a red light 40 m back is not yet
        # required to be stationary.
        #
        # D64: the band used to be 1 m wide, which meant driving *through* a red
        # light registered only the two or three frames spent inside it. The
        # band is now wider, and the crossing itself is a separate, unmissable
        # violation below.
        if (
            obs.distance_to_stop_line is not None
            and 0.0 <= obs.distance_to_stop_line <= RED_LIGHT_STOP_BAND
        ):
            rules["red_light_stop"] = obs.speed < YIELD_SPEED_RED_LIGHT

    # Crossing a stop line while the light is red. Reported separately from the
    # yield rule so that "failed to stop at the line" and "drove through the
    # light" are distinguishable in the profile rather than one blurred number.
    if obs.ran_red_light:
        rules["red_light_run"] = False

    if obs.at_yellow_light and obs.distance_to_stop_line is not None:
        # "Begin deceleration if > 30 m from the stop line". Closer than 30 m
        # the safe action is to clear the intersection, so no rule applies.
        if obs.distance_to_stop_line > YELLOW_LIGHT_DECEL_DISTANCE:
            rules["yellow_light_decelerate"] = obs.is_decelerating

    if obs.nearest_pedestrian_distance is not None:
        # "Pedestrian yield: 5 m". Read as a clearance the ego must maintain
        # unless it is already stopped.
        rules["pedestrian_yield"] = (
            obs.nearest_pedestrian_distance >= PEDESTRIAN_YIELD_DISTANCE
            or obs.speed < YIELD_SPEED_RED_LIGHT
        )

    if obs.approaching_crosswalk:
        rules["crosswalk_speed"] = obs.speed <= CROSSWALK_APPROACH_SPEED

    return rules


class EpisodeCompliance:
    """Aggregates per-frame reports over an episode.

    Chapter 3 requires the dataset's expert to satisfy the metrics as *hard*
    constraints -- "even a single infraction is considered a failure" -- so the
    aggregate carries a pass/fail alongside the usual rates.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.clean_frames = 0
        self.violation_counts: Dict[str, int] = {}
        self.ratio_sums: Dict[str, float] = {}
        self.ratio_maxes: Dict[str, float] = {}
        self.hard_violations = 0

    def add(self, report: ComplianceReport) -> None:
        self.frames += 1
        if report.compliant:
            self.clean_frames += 1
        if report.hard_violation:
            self.hard_violations += 1
        for name in report.violations:
            self.violation_counts[name] = self.violation_counts.get(name, 0) + 1
        for name, ratio in report.ratios.items():
            self.ratio_sums[name] = self.ratio_sums.get(name, 0.0) + ratio
            self.ratio_maxes[name] = max(self.ratio_maxes.get(name, 0.0), ratio)

    @property
    def passed(self) -> bool:
        """True only if the episode is free of infractions, per Chapter 3."""
        return self.frames > 0 and not self.violation_counts

    def summary(self) -> Dict[str, object]:
        n = max(self.frames, 1)
        return {
            "frames": self.frames,
            "passed": self.passed,
            "hard_violation_frames": self.hard_violations,
            "violation_rate": {k: v / n for k, v in sorted(self.violation_counts.items())},
            "mean_ratio": {k: v / n for k, v in sorted(self.ratio_sums.items())},
            "max_ratio": dict(sorted(self.ratio_maxes.items())),
            # Fraction of frames with no infraction of any kind. Bounded to
            # [0, 1] by construction, unlike a per-violation count which can
            # exceed one violation per frame.
            "compliance_score": self.clean_frames / n,
        }
