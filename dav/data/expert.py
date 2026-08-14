"""Rule-based defensive expert agent.

Chapter 3: "The ego agent driving the vehicle is a custom rule-based expert
agent which had the metrics defined above as explicit constraints to ensure the
defensive nature of the dataset. The constraints were enforced as hard
constraints which means that even a single infraction is considered a failure."

The intent is a driver built so that Table 1 holds on every frame, rather than
a good driver that happens to be safe.  Control is a cascade of speed ceilings,
the tightest of which wins, followed by acceleration and jerk limiting so the
commanded change stays inside the comfort envelope.

**It did not hold.**  Written as though correct by construction, the agent was
measured clean in 15.1% of frames.  Roughly forty defects (DOCUMENTATION.md
D23-D67) were needed to reach the 70% acceptance rule, and several were the
comfort limiters themselves preventing safe behaviour -- a jerk limit that
gated emergency braking, a steering cap that produced understeer, a stopping
demand that omitted the distance left.  Do not assume this file is correct
because it reads as though it enforces the rules; check ``trace_<episode>.csv``.

The agent needs privileged simulator state (traffic light status, other actors'
transforms, the map graph).  That is fine and intended: it produces the
training data, it is never deployed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..metrics.evaluator import EgoObservation
from ..metrics.thresholds import (
    BRAKE_DEFENSIVE_TARGET,
    BRAKE_MAX_EMERGENCY,
    BRAKE_NORMAL_TARGET,
    CROSSWALK_APPROACH_SPEED,
    JERK_LAT_ACCEPTABLE,
    JERK_LON_ACCEPTABLE,
    LAT_ACCEL_MAX_COMFORTABLE,
    LON_ACCEL_MAX_COMFORTABLE,
    PEDESTRIAN_YIELD_DISTANCE,
    YELLOW_LIGHT_DECEL_DISTANCE,
    DrivingContext,
    following_gap_for,
    is_heavy_vehicle,
    speed_limit_for,
)

try:  # pragma: no cover - exercised only with a CARLA install present
    import carla
except ImportError:  # pragma: no cover
    carla = None


# --- Longitudinal loop tuning (D23) ----------------------------------------
#: Time constant of the acceleration low-pass, seconds. Long enough to reject
#: the solver's per-substep noise, short enough that a real braking event is
#: not delayed into a late stop.
ACCEL_FILTER_TAU = 0.30
#: Time constant of the *reported* acceleration, seconds. Slower than the
#: control filter on purpose: the controller needs a responsive estimate, while
#: Table 1's acceleration and jerk rows are comfort limits and comfort is a
#: property of the motion a passenger integrates over about a second.
COMFORT_FILTER_TAU = 1.00
#: Samples in the median prefilter (0.25 s). Odd, and long enough to reject a
#: two-sample gearshift impulse.
MEDIAN_WINDOW = 5
#: Velocity-form PI gains, in pedal units per (m/s^2).
PEDAL_KP = 0.06
PEDAL_KI = 0.55
#: Acceleration produced per unit of pedal travel, m/s^2. Measured by
#: regressing achieved acceleration on pedal position over a Town01 episode
#: (n=322 cruising frames): 3.83. Used only to convert the jerk limit into a
#: pedal slew rate, so an approximate figure is sufficient.
PEDAL_ACCEL_GAIN = 3.83
#: Pedal units per second.
#:
#: Sized at 3x the rate needed to *follow* a jerk-limited reference
#: (0.85 * JERK_LON / GAIN = 0.133/s). The reference acceleration produced by
#: ``_limit_longitudinal`` is already jerk-limited, and that is what bounds the
#: jerk a passenger feels; this cap is a backstop against the PI's transient
#: correction, not the primary mechanism. Setting it *equal* to the reference
#: rate left the loop no authority for error correction at all -- the ego
#: closed on a stopped car from 20.8 m while accelerating, because it could not
#: track its own falling target. Setting it by eye at 1.2 in the first version
#: was the opposite error, a 9x jerk overshoot.
PEDAL_MAX_RATE = 1.2 * 0.85 * JERK_LON_ACCEPTABLE / PEDAL_ACCEL_GAIN
#: Emergency escape hatch: when a genuine hazard demands it, the pedal may slew
#: fast enough to stop the car. Comfort is worth less than not hitting things.
PEDAL_EMERGENCY_RATE = 2.0
#: Jerk allowed on the *commanded* acceleration during an emergency, m/s^3.
#: Well above the Table 1 comfort row, which is the point: see D33.
JERK_EMERGENCY = 12.0
#: Fraction of the Table 1 jerk row spent on the *planned* acceleration change
#: (D46). Deliberately well below the 0.85 used for other rows: the achieved
#: jerk is the planned change plus whatever the PI adds correcting tracking
#: error, so planning at the limit guarantees exceeding it. Measured evidence
#: that the residual is real motion rather than noise: jerk sign flips in only
#: 8.0% of frames (white noise is ~50%), lag-1 autocorrelation +0.92, and
#: violations arrive in runs of 0.3 s and longer. Filtering harder would hide
#: genuine jerk, so the vehicle has to be genuinely smoother instead.
JERK_PLAN_MARGIN = 0.5

# --- Planning vs judging horizons (D25) ------------------------------------
#: Distance within which the *evaluator* asks whether a crosswalk applies.
#: The expert must plan beyond this; see ``_plan_lookahead``.
CROSSWALK_JUDGE_LOOKAHEAD = 15.0
#: How far off the ego's path a pedestrian may stand and still trigger the
#: yield rule, metres (D39). Wide enough to cover the lane and a step off the
#: kerb, narrow enough to exclude people simply walking along the pavement.
PEDESTRIAN_LATERAL_TOLERANCE = 1.5
#: How far off the ego's path a crossing may sit and still count, metres.
#: Roughly one lane half-width plus a margin -- enough to catch the crossing
#: the ego will actually drive over, not the one on the cross street.
CROSSWALK_LATERAL_TOLERANCE = 4.0
#: How far off the path a speed-limit sign may sit and still apply, metres (D47).
#: Signs sit at the roadside, so this is wider than the lane.
SPEED_SIGN_LATERAL_TOLERANCE = 6.0
#: Driver reaction allowance folded into the planning horizon, seconds.
REACTION_TIME = 1.0
#: Speed cap when approaching an unsignalled junction, m/s (~29 km/h). Higher
#: than a marked pedestrian crossing; Table 1 does not name a figure, so this
#: is a judgement call recorded as part of D27.
JUNCTION_APPROACH_SPEED = 8.0
#: Fraction of the dry lateral-acceleration limit usable on a wet road (D55).
WET_GRIP_FACTOR = 0.70

# --- Crossing-conflict prediction (D30) -------------------------------------
#: Only vehicles within this radius are considered, metres.
CONFLICT_SCAN_RADIUS = 40.0
#: How far ahead to extrapolate, seconds.
CONFLICT_HORIZON = 5.0
#: Predicted miss distance at which a conflict is declared, metres.
#:
#: This must be smaller than a lane width (3.5 m in CARLA) or the check fires
#: on traffic that is merely *passing* rather than crossing: oncoming vehicles
#: in the opposite lane pass within one lane width by definition. An initial
#: value of 4.5 did exactly that -- in Town10HD the ego yielded to essentially
#: every vehicle on the road, travelled 3.9 m in 20 s, and scored 95.2%
#: compliance by standing still. Two vehicle half-widths plus a small margin.
CONFLICT_CLEARANCE = 3.0
#: Vehicles slower than this cannot create a crossing conflict; treating parked
#: cars as hazards deadlocks the ego.
CONFLICT_MIN_SPEED = 0.5
#: Stop this far short of the predicted conflict point, metres.
CONFLICT_BUFFER = 5.0
#: Bumper-to-bumper gap to keep when stopped behind another vehicle, metres (D32).
STANDSTILL_GAP = 5.0
#: Clearance to leave in front when stopping, metres (D61). The bumper gap is
#: estimated from bounding-box extents, which under-reports for long vehicles
#: like buses -- the ego "stopped" with 0.21 m showing and still made contact.
CONTACT_BUFFER = 1.5
#: Stop this far short of the stop line, metres (D64).
STOP_LINE_MARGIN = 2.0
#: How far off the ego's path a stop line may sit and still govern it, m (D66).
TRAFFIC_LIGHT_LATERAL_TOLERANCE = 3.0
#: How far ahead to project the closing rate when sizing the following gap,
#: seconds (D34). Roughly the actuator's settling time.
GAP_PREDICTION_TIME = 1.5

# --- Lateral geometry (D26) -------------------------------------------------
#: Approximate wheelbase of CARLA's default vehicles, metres. Used to turn a
#: lateral-acceleration limit into a steering-angle limit via the bicycle model.
WHEELBASE = 2.9
#: Steering angle at full lock, radians. CARLA's steer input is normalised to
#: [-1, 1] against this.
MAX_STEER_ANGLE = math.radians(70.0)
#: Floor on the per-tick steer rate, so the vehicle can still negotiate a tight
#: junction at low speed rather than understeering into the opposing lane.
MIN_STEER_RATE = 0.01
#: Pure-pursuit lookahead: floor in metres and seconds-ahead scaling (D41).
#: Shortening this to 3.0/1.0 was tried twice and rejected twice. The second
#: time the route cursor bug (D42) was already fixed, so the geometry itself is
#: the problem: a short lookahead tracks tight corners better but oscillates at
#: speed, and the same seed that ran 3764 frames at 4.0/1.2 collided at 980.
LOOKAHEAD_MIN = 4.0
LOOKAHEAD_TIME = 1.2
#: Route points searched behind / ahead of the cursor when measuring lateral
#: offset (D40). The route is spaced at 2 m.
ROUTE_SEARCH_BACK = 5
ROUTE_SEARCH_FORWARD = 15
#: Anti-windup bound on the accumulated lateral error, metre-seconds (D28).
CROSSTRACK_INTEGRAL_LIMIT = 1.0
#: Per-tick leak, so the accumulator forgets a disturbance that has passed
#: rather than holding a correction into the next stretch of road.
CROSSTRACK_INTEGRAL_LEAK = 0.998
#: Faster decay while inside a junction, where lane centre is ill-defined.
JUNCTION_INTEGRAL_DECAY = 0.95
#: Lateral error beyond which the steer rate limit is relaxed, metres (D43).
#: Above the 0.30 m Table 1 row: by this point the ego is already violating it
#: and the priority is getting back, not staying comfortable.
CROSSTRACK_RECOVERY_THRESHOLD = 0.5
#: Steer units per tick allowed while recovering (D43).
STEER_RECOVERY_RATE = 0.05

# --- Route curvature anticipation (D43) -------------------------------------
#: Route distance scanned for an upcoming turn: floor in metres and a
#: seconds-ahead term, so the ego starts slowing with room to spare.
CURVE_SCAN_MIN = 12.0
CURVE_SCAN_TIME = 2.5
#: Time constant for *releasing* the curvature ceiling once the turn is past,
#: seconds (D45). Only the rising edge is filtered; falling is instant.
CURVE_RELEASE_TAU = 1.20

# --- Outer speed loop -------------------------------------------------------
#: Time constant of the speed -> acceleration outer loop, seconds. The inner
#: loop is deliberately slow (the jerk row caps torque slew), so a fast outer
#: loop simply commands acceleration the vehicle cannot deliver yet and
#: overshoots the speed limit while the inner loop catches up.
SPEED_LOOP_TAU = 1.5


def _closer(current: Optional[float], candidate: float) -> float:
    """Nearest of two stopping distances, ignoring None (D65)."""
    return candidate if current is None else min(current, candidate)


def _is_finite_location(location) -> bool:
    """Guard against CARLA handing back a non-finite actor transform.

    D44: a walker whose transform goes NaN -- which CARLA does when an actor is
    destroyed mid-tick or falls out of the world -- poisoned the pedestrian
    yield rule permanently. ``nan >= 5.0`` is False and ``nan < speed`` is
    False, so *both* branches of "clear of pedestrians OR stopped" evaluate
    False and every subsequent frame is scored as a violation. Measured: 142
    consecutive violated frames from the moment it appeared to the end of the
    episode, identical in every run because it is not a driving behaviour at
    all.

    NaN does not raise, it silently answers "no" to every question. Anything
    read from the simulator and then compared against a threshold needs this.
    """
    return (
        math.isfinite(location.x)
        and math.isfinite(location.y)
        and math.isfinite(location.z)
    )


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


@dataclass
class ExpertState:
    """Carried between ticks so acceleration and jerk can be rate-limited."""

    target_speed: float = 0.0
    commanded_accel: float = 0.0
    prev_speed: float = 0.0
    prev_accel_vec: Tuple[float, float] = (0.0, 0.0)
    prev_steer: float = 0.0
    #: Pedal command in [-1, 1]: positive throttle, negative brake. Held across
    #: ticks because the longitudinal loop is incremental (see D23).
    pedal: float = 0.0
    #: Previous acceleration error, for the velocity-form PI.
    prev_accel_error: float = 0.0
    #: Low-pass filtered longitudinal / lateral acceleration, control bandwidth (D24).
    filt_lon_accel: float = 0.0
    filt_lat_accel: float = 0.0
    filt_initialised: bool = False
    #: Comfort-bandwidth estimates, reported to the evaluator (D24).
    comfort_lon_accel: float = 0.0
    comfort_lat_accel: float = 0.0
    #: Recent raw samples, for the median prefilter that rejects gear-shift steps.
    raw_lon_window: List[float] = field(default_factory=list)
    raw_lat_window: List[float] = field(default_factory=list)
    #: Accumulated lateral error, for removing steady-state lane offset (D28).
    crosstrack_integral: float = 0.0
    #: Set when an obstacle is close enough that the *gap*, not the required
    #: deceleration, is what makes the situation urgent (D60).
    proximity_urgent: bool = False
    #: Distance to the nearest obstacle the ego must stop *before* -- a lead
    #: vehicle or a red stop line -- less its safety buffer (D65). None when
    #: nothing demands a stop.
    stop_room: Optional[float] = None
    #: Previous frame's signed distance to a red light's stop line (D64), so a
    #: crossing can be detected as a sign change rather than a proximity test.
    stop_line_distance: Optional[float] = None
    #: Held curvature speed ceiling; falls instantly, rises on a filter (D45).
    curve_ceiling: float = float("inf")
    #: Steering the path follower asked for before the comfort cap (D35). Feeds
    #: back into the speed target so a demanded turn becomes feasible instead of
    #: being silently clipped into understeer.
    desired_steer: float = 0.0
    #: Rate-limited target speed, so a change of binding constraint does not
    #: step the setpoint and inject a jerk transient.
    smoothed_target: float = 0.0


class DefensiveExpert:
    """Waypoint-following expert with Table 1 as hard constraints.

    Parameters
    ----------
    world, vehicle
        CARLA handles.
    route
        List of ``carla.Location`` waypoints to follow.
    dt
        Fixed simulation timestep in seconds.
    safety_margin
        Fraction of each threshold the agent actually targets. Aiming at
        exactly 1.0x threshold guarantees occasional overshoot from
        discretisation, and one overshoot fails the whole episode, so the
        expert plans against ``safety_margin`` x threshold. 0.85 leaves enough
        headroom to survive a control step without being so timid that the data
        stops resembling driving.
    """

    def __init__(
        self,
        world,
        vehicle,
        route: List,
        dt: float = 0.05,
        safety_margin: float = 0.85,
    ) -> None:
        self.world = world
        self.vehicle = vehicle
        self.route = route
        self.dt = dt
        self.margin = safety_margin
        self.state = ExpertState()
        self.map = world.get_map()
        self._route_index = 0
        #: Most recent driving context, so grip-dependent limits can see the
        #: weather without re-deriving it per call (D55).
        self._context_cache: Optional[DrivingContext] = None
        #: Traffic light actors, cached per episode (D66); they do not change.
        self._traffic_lights = None

        # Lateral controller gains. Deliberately soft: an aggressive steering
        # controller is the main way a rule-based expert breaks the lateral
        # acceleration and lateral jerk rows of Table 1.
        self.k_heading = 0.8
        # D62: raised from 0.35. At a 2.4 m offset and 5 m/s the old gain asked
        # for only 0.157 of steer -- and D35's speed ceiling derived from that
        # same figure came out at exactly 5 m/s, so the ego held speed, held
        # steer, and drifted wide into a pole. The loop has to *want* more
        # steer before the speed ceiling will slow it enough to deliver it.
        # Safe to raise now that D42 fixed the route cursor; the earlier
        # attempt at 0.75 was destabilised by that bug, not by the gain.
        self.k_crosstrack = 0.85
        self.k_crosstrack_integral = 0.025
        self.max_steer_rate = 0.15  # per tick, limits lateral jerk

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def build_context(self) -> DrivingContext:
        weather = self.world.get_weather()
        transform = self.vehicle.get_transform()
        waypoint = self.map.get_waypoint(transform.location)

        speed_limit_ms = self.vehicle.get_speed_limit() / 3.6
        if speed_limit_ms <= 0:
            speed_limit_ms = 30.0 / 3.6  # CARLA returns 0 before the first tick

        # CARLA exposes wetness and precipitation as 0-100 scalars.
        wet = weather.wetness > 10.0 or weather.precipitation > 10.0
        # "Low visibility" from heavy precipitation or fog.
        low_visibility = weather.precipitation > 40.0 or weather.fog_density > 20.0
        # Sun below the horizon is night.
        night = weather.sun_altitude_angle < 0.0

        lead, lead_distance = self._find_lead_vehicle()
        heavy = bool(lead is not None and is_heavy_vehicle(lead.type_id))

        return DrivingContext(
            speed_limit_ms=speed_limit_ms,
            wet=wet,
            low_visibility=low_visibility,
            night=night,
            lead_is_heavy_vehicle=heavy,
        )

    # ------------------------------------------------------------------
    # Perception (privileged)
    # ------------------------------------------------------------------

    def _find_lead_vehicle(self, max_distance: float = 60.0):
        """Nearest vehicle ahead in the ego's lane, and its bumper distance."""
        ego_tf = self.vehicle.get_transform()
        ego_wp = self.map.get_waypoint(ego_tf.location)
        forward = ego_tf.get_forward_vector()
        ego_extent = self.vehicle.bounding_box.extent.x

        best = None
        best_distance = max_distance

        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.vehicle.id:
                continue
            other_tf = actor.get_transform()
            if not _is_finite_location(other_tf.location):
                continue  # D44
            delta = other_tf.location - ego_tf.location
            forward_distance = delta.x * forward.x + delta.y * forward.y
            if forward_distance <= 0 or forward_distance > max_distance:
                continue

            # Same lane check via the road graph, which is more reliable than a
            # lateral-offset test on curved roads.
            other_wp = self.map.get_waypoint(other_tf.location)
            if (
                other_wp.road_id != ego_wp.road_id
                or other_wp.lane_id != ego_wp.lane_id
            ):
                # Allow the vehicle immediately ahead across a road-segment
                # boundary, which is common at junction approaches.
                lateral = abs(delta.x * -forward.y + delta.y * forward.x)
                if lateral > 1.75:
                    continue

            bumper_gap = forward_distance - ego_extent - actor.bounding_box.extent.x
            if bumper_gap < best_distance:
                best_distance = max(bumper_gap, 0.0)
                best = actor

        return best, (best_distance if best is not None else None)

    @staticmethod
    def closest_point_of_approach(
        px: float, py: float, vx: float, vy: float
    ) -> Optional[Tuple[float, float]]:
        """``(time, miss distance)`` of closest approach, or None if separating.

        ``p`` is the other vehicle's position relative to the ego and ``v`` its
        velocity relative to the ego, both in the world frame. Pure geometry,
        kept separate from the actor scan so it can be tested without a
        simulator. See D30.
        """
        closing = vx * vx + vy * vy
        if closing < 1e-6:
            return None  # no relative motion: parallel traffic at equal speed
        t = -(px * vx + py * vy) / closing
        if t <= 0.0:
            return None  # already at or past the closest point: separating
        return t, math.hypot(px + vx * t, py + vy * t)

    def _conflict_distance(self) -> Optional[float]:
        """Distance along the ego's path to the nearest predicted conflict.

        D30: ``_find_lead_vehicle`` matches on ``road_id``/``lane_id``, so it
        sees only traffic in the ego's own lane. Inside a junction, crossing
        traffic has a different road and lane and a large lateral offset, and
        was therefore invisible -- with 40 vehicles spawned, a lead vehicle was
        detected in 2.2% of frames, and three consecutive Town01 attempts ended
        in a junction collision at ~6.4 m/s with no hazard ever registered.

        This is lane-agnostic: it extrapolates both vehicles at constant
        velocity and finds the closest point of approach. Returns the distance
        the ego travels before that point, or None when nothing conflicts.
        """
        ego_tf = self.vehicle.get_transform()
        ego_v = self.vehicle.get_velocity()
        ego_speed = math.sqrt(ego_v.x**2 + ego_v.y**2)
        forward = ego_tf.get_forward_vector()

        nearest: Optional[float] = None
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.vehicle.id:
                continue
            other_tf = actor.get_transform()
            if not _is_finite_location(other_tf.location):
                continue  # D44
            px = other_tf.location.x - ego_tf.location.x
            py = other_tf.location.y - ego_tf.location.y
            if px * px + py * py > CONFLICT_SCAN_RADIUS**2:
                continue

            other_v = actor.get_velocity()
            # A stationary vehicle cannot create a *crossing* conflict, and
            # treating parked cars as hazards deadlocks the ego permanently.
            # Obstacles in the ego's own lane remain the lead-vehicle case.
            if math.sqrt(other_v.x**2 + other_v.y**2) < CONFLICT_MIN_SPEED:
                continue

            cpa = self.closest_point_of_approach(
                px, py, other_v.x - ego_v.x, other_v.y - ego_v.y
            )
            if cpa is None:
                continue
            t, miss = cpa
            if t > CONFLICT_HORIZON or miss > CONFLICT_CLEARANCE:
                continue

            # Only yield to conflicts ahead of us; being overtaken is not ours
            # to solve by braking.
            if px * forward.x + py * forward.y <= 0.0:
                continue

            travel = ego_speed * t
            nearest = travel if nearest is None else min(nearest, travel)
        return nearest

    def _nearest_pedestrian(self, max_distance: float = 30.0) -> Optional[float]:
        ego_tf = self.vehicle.get_transform()
        forward = ego_tf.get_forward_vector()
        nearest = None
        for walker in self.world.get_actors().filter("walker.pedestrian.*"):
            walker_location = walker.get_transform().location
            if not _is_finite_location(walker_location):
                continue  # D44
            delta = walker_location - ego_tf.location
            distance = math.sqrt(delta.x**2 + delta.y**2)
            if distance > max_distance:
                continue
            # Only pedestrians in front matter for the yield rule.
            if delta.x * forward.x + delta.y * forward.y <= 0:
                continue
            # ...and only those on or near the ego's path. D39: without a
            # lateral test this counts pedestrians standing on the pavement
            # beside the road, and the 5 m yield clearance is then violated
            # almost continuously in a populated town -- measured at 36.9% of
            # frames in Town01 with 20 walkers. The same detector feeds the
            # evaluator, so the metric inherited the error too.
            lateral = abs(-delta.x * forward.y + delta.y * forward.x)
            if lateral > PEDESTRIAN_LATERAL_TOLERANCE:
                continue
            nearest = distance if nearest is None else min(nearest, distance)
        return nearest

    def _upcoming_traffic_light(self, lookahead: float):
        """Nearest light governing the ego's lane ahead: ``(state, distance)``.

        D66: ``vehicle.get_traffic_light()`` only returns a light once the ego
        is inside its trigger volume, which is a few metres before the stop
        line. Measured: the ``red_light`` constraint first appeared **7 frames
        (0.35 s) before the crossing** at 7-8 m/s -- 2.5 m of road. The expert
        commanded a full stop and braked hard, and still crossed, because no
        controller can stop in that distance. It was not a control failure; the
        light was invisible until it was too late.

        Scanning the light actors directly gives the anticipation the stopping
        profile needs, exactly as D47 does for speed-limit signs.
        """
        tf = self.vehicle.get_transform()
        forward = tf.get_forward_vector()
        ego_wp = self.map.get_waypoint(tf.location)

        if self._traffic_lights is None:
            self._traffic_lights = list(
                self.world.get_actors().filter("traffic.traffic_light*")
            )

        best = None
        for light in self._traffic_lights:
            try:
                stops = light.get_stop_waypoints()
            except RuntimeError:
                continue
            for w in stops:
                loc = w.transform.location
                if not _is_finite_location(loc):
                    continue
                dx = loc.x - tf.location.x
                dy = loc.y - tf.location.y
                ahead = dx * forward.x + dy * forward.y
                if ahead <= 0.0 or ahead > lookahead:
                    continue
                # Must govern our lane: same road and lane where the road graph
                # agrees, otherwise a tight lateral tolerance.
                same_lane = (
                    w.road_id == ego_wp.road_id and w.lane_id == ego_wp.lane_id
                )
                lateral = abs(-dx * forward.y + dy * forward.x)
                if not same_lane and lateral > TRAFFIC_LIGHT_LATERAL_TOLERANCE:
                    continue
                if best is None or ahead < best[1]:
                    best = (str(light.get_state()), ahead)
        return best

    def _traffic_light_state(self) -> Tuple[Optional[str], Optional[float]]:
        """``(state, signed distance to the stop line)``.

        D64: the distance is *signed* -- positive while the line is ahead,
        negative once the ego has crossed it. It used to be
        ``Location.distance()``, which is Euclidean and therefore grows again
        on the far side of the line. Two things followed from that:

          * the expert's stopping ceiling is ``sqrt(2*a*d)``, so as the ego
            drove *through* a junction the permitted speed *increased* with
            distance past the line -- the ceiling actively encouraged running
            the light;
          * the evaluator only asked for a stop within 1 m of the line, so
            crossing at speed registered the two or three frames spent inside
            that band and nothing else. A red light run scored 0.23%.

        The stop waypoint is also chosen by lane rather than by proximity:
        ``get_stop_waypoints`` returns one per lane the light controls, and the
        nearest can belong to a neighbouring lane.
        """
        # D66: prefer the forward scan, which sees the light in time to stop.
        speed = math.sqrt(
            self.vehicle.get_velocity().x ** 2 + self.vehicle.get_velocity().y ** 2
        )
        upcoming = self._upcoming_traffic_light(
            self._plan_lookahead(speed, 0.0) + STOP_LINE_MARGIN
        )
        if upcoming is not None:
            return upcoming

        light = self.vehicle.get_traffic_light()
        if light is None:
            return None, None
        state = str(light.get_state())  # e.g. "Red"
        waypoints = light.get_stop_waypoints()
        if not waypoints:
            return state, None

        tf = self.vehicle.get_transform()
        ego_wp = self.world.get_map().get_waypoint(tf.location)
        forward = tf.get_forward_vector()

        # Prefer the stop waypoint on the ego's own lane; fall back to the one
        # most nearly straight ahead.
        candidates = [
            w for w in waypoints
            if w.road_id == ego_wp.road_id and w.lane_id == ego_wp.lane_id
        ] or list(waypoints)

        best = None
        for w in candidates:
            dx = w.transform.location.x - tf.location.x
            dy = w.transform.location.y - tf.location.y
            along = dx * forward.x + dy * forward.y  # signed: + is ahead
            if best is None or abs(along) < abs(best):
                best = along
        return state, best

    # ------------------------------------------------------------------
    # Speed planning: every ceiling, tightest wins
    # ------------------------------------------------------------------

    def _target_speed(self, ctx: DrivingContext, speed: float) -> Tuple[float, str]:
        """Returns (target speed in m/s, name of the binding constraint)."""
        ceilings: list[tuple[float, str]] = []

        # Row 1: context-adjusted speed limit.
        # NOTE: an extra adverse-weather speed margin was tried here and
        # removed. The reasoning -- that Table 1's weather factors and the
        # expert's own margin compound badly -- was contradicted by the pilot:
        # the heaviest weather scored *highest* (MidRainyNoon 90.2%,
        # HardRainNoon 86.2%), because once precipitation crosses the
        # low-visibility threshold the ceiling drops to 0.70x and the ego
        # simply drives slowly. Only the narrow "wet but still fast" band
        # (SoftRain) was marginal, and it passed on a reseed at 75.4%. Slowing
        # the expert further would have cost data quality to fix a problem that
        # was not there.
        # Cleared every tick: the flag is only *set* when a lead vehicle is
        # within the standstill gap, so without this it would latch on the
        # first close pass and leave the expert in emergency mode for the rest
        # of the episode.
        self.state.proximity_urgent = False
        self.state.stop_room = None

        ceilings.append((self.margin * speed_limit_for(ctx), "speed_limit"))

        # D47: be at the *next* limit by the time the sign passes, not after.
        upcoming = self._upcoming_speed_limit(
            max(CURVE_SCAN_MIN, speed * CURVE_SCAN_TIME)
        )
        if upcoming is not None:
            limit, distance = upcoming
            target = self.margin * limit
            if target < speed:
                decel = 0.5 * self.margin * BRAKE_DEFENSIVE_TARGET
                ceilings.append(
                    (
                        math.sqrt(target * target + 2.0 * decel * distance),
                        "speed_limit_ahead",
                    )
                )

        # Row 2: following distance. To hold an N-second gap at distance d, the
        # ego may travel at most d / N.
        lead, lead_distance = self._find_lead_vehicle()
        if lead_distance is not None:
            required_gap = following_gap_for(ctx)
            lead_v = lead.get_velocity() if lead is not None else None
            lead_speed = (
                math.sqrt(lead_v.x**2 + lead_v.y**2) if lead_v is not None else 0.0
            )

            # D34: react to the gap the ego is *about* to have, not the one it
            # has. The plain d/N rule is a position feedback with no lead term,
            # so when the gap is closing the target only falls once the gap has
            # already shrunk -- and by then the jerk-limited actuator cannot
            # catch up. 79% of the residual following-gap violations occurred
            # while the ego was already braking, which is the signature of lag
            # rather than of driving too fast.
            closing = speed - lead_speed
            predicted = max(lead_distance - closing * GAP_PREDICTION_TIME, 0.0)
            ceilings.append(
                (self.margin * predicted / required_gap, "following_distance")
            )

            # D32: the time-gap rule above is a *steady-state* target. It says
            # nothing about the lead's speed or about the ego's own limited
            # braking, so when the lead decelerates the gap collapses faster
            # than the jerk row lets the ego respond -- measured as a front
            # impact with the vehicle ahead and a 22.4% following-gap violation
            # rate. This adds the kinematic constraint: the ego must be able to
            # match the lead's speed before reaching it.
            #
            # v^2 = v_lead^2 + 2*a*d, with the usable distance reduced by the
            # standstill gap and the deceleration halved as a cheap allowance
            # for the time the jerk limit takes to build the brake.
            # D60: urgency by proximity, not only by deceleration demand.
            #
            # ``_accel_to_pedals`` escalates the pedal slew when the *demanded*
            # deceleration is large. At 1 m/s with an obstacle 0.6 m away the
            # demand is only -0.67 m/s^2 -- below the threshold -- so the pedal
            # bled off at the comfort rate and the ego crept into the back of a
            # stationary bus with the brake never applied. The demand is small
            # because the ego is already slow; the situation is urgent anyway.
            self.state.proximity_urgent = lead_distance < STANDSTILL_GAP
            self.state.stop_room = _closer(
                self.state.stop_room, lead_distance - CONTACT_BUFFER
            )
            usable = max(lead_distance - STANDSTILL_GAP, 0.0)
            decel = 0.5 * self.margin * BRAKE_DEFENSIVE_TARGET
            ceilings.append(
                (math.sqrt(lead_speed**2 + 2.0 * decel * usable), "lead_kinematic")
            )

        # D35: limit speed by the steering the controller is actually asking
        # for. ``_upcoming_curve_radius`` samples the road graph on a fixed
        # 25 m lookahead, which does not capture how sharply the *route* turns
        # through a junction. The steering demand does, because it is what the
        # path follower computed. a_lat = v^2 * curvature, so invert it.
        desired = abs(self.state.desired_steer)
        if desired > 1e-3:
            curvature = math.tan(desired * MAX_STEER_ANGLE) / WHEELBASE
            if curvature > 1e-4:
                ceilings.append(
                    (
                        math.sqrt(self._lateral_limit() / curvature),
                        "steer_demand",
                    )
                )

        # D43: read the turn off the route ahead rather than waiting for the
        # steering demand to reveal it. The distance scanned is the room needed
        # to bleed off speed, so the ego is already slow when the turn arrives.
        # D45: the *magnitude* of the max-curvature ceiling is right -- it slows
        # the ego as soon as a tight turn is anywhere in the window, which is
        # what makes the turn trackable. Its problem was purely that it stepped
        # when the tightest point left the window, and the ego accelerated into
        # the step (65% of the residual jerk violations were while accelerating).
        #
        # Replacing it with a smooth sqrt(v^2 + 2ad) profile was worse, because
        # that profile assumes a deceleration the jerk row will not deliver and
        # the ego arrives at the turn too fast: lane offset went 6.2% -> 39.5%.
        #
        # So: keep the conservative magnitude, fix only the shape. The ceiling
        # falls instantly -- a speed limit that arrives late is a safety
        # failure -- and rises on a filter, which is where the jerk was.
        curvature = self._route_curvature_ahead(
            max(CURVE_SCAN_MIN, speed * CURVE_SCAN_TIME)
        )
        if curvature > 1e-4:
            raw_ceiling = math.sqrt(
                self._lateral_limit() / curvature
            )
        else:
            raw_ceiling = float("inf")
        held = self.state.curve_ceiling
        if raw_ceiling <= held:
            self.state.curve_ceiling = raw_ceiling
        else:
            alpha = self.dt / (CURVE_RELEASE_TAU + self.dt)
            self.state.curve_ceiling = held + alpha * (raw_ceiling - held)
        if math.isfinite(self.state.curve_ceiling):
            ceilings.append((self.state.curve_ceiling, "route_curvature"))

        # Row 4: lateral acceleration limits speed through curves. For radius R,
        # v_max = sqrt(a_lat_max * R).
        radius = self._upcoming_curve_radius()
        if radius is not None and radius > 1.0:
            v_curve = math.sqrt(self._lateral_limit() * radius)
            ceilings.append((v_curve, "curve"))

        # Row 6: traffic lights and stop signs.
        light_state, light_distance = self._traffic_light_state()
        if light_state is not None and light_distance is not None:
            if "Red" in light_state and light_distance > 0.0:
                # D64: only while the line is ahead. Past it, stopping dead in
                # the middle of a junction is worse than clearing it. The
                # halved deceleration matches D32: the jerk row will not
                # deliver the full rate on demand, so plan for what it will.
                room = max(light_distance - STOP_LINE_MARGIN, 0.0)
                decel = 0.5 * self.margin * BRAKE_DEFENSIVE_TARGET
                ceilings.append((math.sqrt(2.0 * decel * room), "red_light"))
                # D65: a stop line is a hard stop like any other. Without this
                # the speed loop only ever asked for the proportional term, the
                # jerk budget took ~2.4 s to ramp it to -3 m/s^2, and the ego
                # coasted 10.8 m through the light at 4.4 m/s with the target
                # sitting at 0 the whole time.
                self.state.stop_room = _closer(self.state.stop_room, room)
            elif "Yellow" in light_state:
                # "Begin deceleration if > 30 m from the stop line"; inside 30 m
                # the safe action is to clear the intersection.
                if light_distance > YELLOW_LIGHT_DECEL_DISTANCE:
                    ceilings.append(
                        (self._stopping_profile(light_distance), "yellow_light")
                    )

        # Row 6: pedestrian yield.
        pedestrian = self._nearest_pedestrian()
        if pedestrian is not None:
            ceilings.append(
                (
                    self._stopping_profile(
                        max(pedestrian - PEDESTRIAN_YIELD_DISTANCE, 0.0)
                    ),
                    "pedestrian",
                )
            )

        # Row 6: crosswalk approach speed.
        #
        # D25: the planning horizon must exceed the horizon the rule is judged
        # on. The evaluator asks "is a crosswalk within JUDGE_LOOKAHEAD, and is
        # the speed above the cap?" -- an instantaneous test. If the expert
        # only starts slowing at that same distance, then every frame of the
        # (physically unavoidable) deceleration is scored as a violation. The
        # expert therefore looks far enough ahead to *arrive* at the cap, using
        # the distance needed to bleed off the speed at the defensive rate.
        crosswalk_cap = self.margin * CROSSWALK_APPROACH_SPEED
        if self._approaching_crosswalk(
            lookahead=self._plan_lookahead(speed, crosswalk_cap)
        ):
            ceilings.append((crosswalk_cap, "crosswalk"))

        # Row 6 / give-way: yield to predicted crossing conflicts (D30).
        conflict = self._conflict_distance()
        if conflict is not None:
            ceilings.append(
                (self._stopping_profile(max(conflict - CONFLICT_BUFFER, 0.0)), "conflict")
            )

        # Junctions get a speed cap too -- a defensive driver slows for an
        # intersection whether or not a marked crossing is present -- but a
        # higher one than a pedestrian crossing, and so a shorter approach.
        junction_cap = self.margin * JUNCTION_APPROACH_SPEED
        if self._approaching_junction(
            lookahead=self._plan_lookahead(speed, junction_cap)
        ):
            ceilings.append((junction_cap, "junction"))

        # D65: the pedal and jerk escalations key off ``proximity_urgent``,
        # which was set only by the lead-vehicle branch. A stop line close
        # enough to demand hard braking deserves the same treatment, or the
        # reference ramps at the comfort budget while the line goes past.
        if self.state.stop_room is not None and self.state.stop_room < STANDSTILL_GAP:
            self.state.proximity_urgent = True

        target, reason = min(ceilings, key=lambda pair: pair[0])
        return max(target, 0.0), reason

    def _stopping_profile(self, distance: float) -> float:
        """Speed from which the vehicle can stop in ``distance`` at the defensive rate.

        v = sqrt(2 * a * d) with the defensive braking target, not the emergency
        one -- the whole point is to never need the emergency rate.
        """
        if distance <= 0.5:
            return 0.0
        return math.sqrt(2.0 * self.margin * BRAKE_DEFENSIVE_TARGET * distance)

    def _upcoming_curve_radius(self, lookahead: float = 25.0) -> Optional[float]:
        """Radius of the road ahead, from the yaw change over the lookahead."""
        tf = self.vehicle.get_transform()
        wp = self.map.get_waypoint(tf.location)
        ahead = wp.next(lookahead)
        if not ahead:
            return None
        yaw0 = math.radians(wp.transform.rotation.yaw)
        yaw1 = math.radians(ahead[0].transform.rotation.yaw)
        delta = abs(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)))
        if delta < 1e-3:
            return None
        return lookahead / delta

    def _curvature_speed_profile(self, distance: float) -> Optional[float]:
        """Speed ceiling from the turns ahead, as a continuous function.

        D45: taking the *maximum* curvature over a sliding window, as the first
        version of D43 did, is discontinuous in position -- the instant the
        tightest point falls out of the window the ceiling jumps and the ego
        accelerates into the step. Measured: 65% of the remaining longitudinal
        jerk violations happened while *accelerating*, median +0.69 m/s^2, with
        ``route_curvature`` the binding constraint on half of them. The
        anticipation was correct; its shape was not.

        Instead each point ahead contributes the speed the ego may hold *now*
        in order to be at that point's curvature limit when it arrives:

            v_allowed = sqrt(v_curve(p)^2 + 2 * a * d(p))

        and the ceiling is the minimum over the window. A point's constraint
        now relaxes smoothly as it recedes rather than vanishing at the edge.
        """
        if len(self.route) < 3:
            return None
        decel = self.margin * BRAKE_DEFENSIVE_TARGET
        max_lat = self._lateral_limit()

        travelled = 0.0
        ceiling = None
        i = self._route_index
        while i + 2 < len(self.route) and travelled < distance:
            a, b, c = self.route[i], self.route[i + 1], self.route[i + 2]
            first = math.hypot(b.x - a.x, b.y - a.y)
            second = math.hypot(c.x - b.x, c.y - b.y)
            if first < 1e-3 or second < 1e-3:
                i += 1
                continue
            heading_a = math.atan2(b.y - a.y, b.x - a.x)
            heading_b = math.atan2(c.y - b.y, c.x - b.x)
            turn = abs(
                math.atan2(
                    math.sin(heading_b - heading_a), math.cos(heading_b - heading_a)
                )
            )
            curvature = turn / (0.5 * (first + second))
            if curvature > 1e-4:
                at_point = math.sqrt(max_lat / curvature)
                allowed = math.sqrt(at_point * at_point + 2.0 * decel * travelled)
                ceiling = allowed if ceiling is None else min(ceiling, allowed)
            travelled += first
            i += 1
        return ceiling

    def _route_curvature_ahead(self, distance: float) -> float:
        """Largest curvature (1/m) of the route within ``distance`` metres.

        D43: the expert previously discovered a turn only once the steering
        demand had already grown (D35), which is too late. The lateral jerk row
        caps how fast the wheel may turn, so at 1.3 m/s the wheel needed 1.5 s
        to reach the lock a junction required -- and the vehicle drove straight
        out of the corner meanwhile, 3.9 m wide, taking 7.5 s to recover.
        Reading the curvature off the route ahead lets the ego arrive slow
        enough to make the turn inside the comfort envelope.
        """
        if len(self.route) < 3:
            return 0.0
        travelled = 0.0
        worst = 0.0
        i = self._route_index
        while i + 2 < len(self.route) and travelled < distance:
            a, b, c = self.route[i], self.route[i + 1], self.route[i + 2]
            first = math.hypot(b.x - a.x, b.y - a.y)
            second = math.hypot(c.x - b.x, c.y - b.y)
            travelled += first
            if first < 1e-3 or second < 1e-3:
                i += 1
                continue
            # Turn angle between successive segments over the arc length.
            heading_a = math.atan2(b.y - a.y, b.x - a.x)
            heading_b = math.atan2(c.y - b.y, c.x - b.x)
            turn = abs(
                math.atan2(
                    math.sin(heading_b - heading_a), math.cos(heading_b - heading_a)
                )
            )
            worst = max(worst, turn / (0.5 * (first + second)))
            i += 1
        return worst

    def _upcoming_speed_limit(self, lookahead: float):
        """Lowest speed-limit sign ahead on the path: ``(limit m/s, distance)``.

        D47: ``vehicle.get_speed_limit()`` reports the limit where the ego *is*.
        Crossing into a slower zone therefore drops the limit instantly while
        the jerk-limited vehicle needs seconds to comply, and every frame of
        that deceleration is scored as speeding -- measured at 11.2% of frames
        on average, up to 17.2%, all marginal (~1.01x) which is the signature of
        arriving late rather than driving fast. Same shape as D25.
        """
        tf = self.vehicle.get_transform()
        forward = tf.get_forward_vector()
        best = None
        for sign in self.world.get_actors().filter("traffic.speed_limit.*"):
            location = sign.get_transform().location
            if not _is_finite_location(location):
                continue
            dx = location.x - tf.location.x
            dy = location.y - tf.location.y
            ahead = dx * forward.x + dy * forward.y
            if ahead <= 0.0 or ahead > lookahead:
                continue
            if abs(-dx * forward.y + dy * forward.x) > SPEED_SIGN_LATERAL_TOLERANCE:
                continue
            try:
                limit = float(sign.type_id.rsplit(".", 1)[-1]) / 3.6
            except ValueError:
                continue
            if best is None or limit < best[0]:
                best = (limit, ahead)
        return best

    def _lateral_limit(self) -> float:
        """Usable lateral acceleration, reduced on a wet road (D55).

        Table 1 scales the *speed* ceiling for wet conditions but says nothing
        about cornering, and the expert treated its lateral limit as a constant
        -- so it cornered as though dry, understeered on a low-friction surface
        and left the lane. Measured on Town01/WetNoon: ``lane_departure``
        violated in 50.7% of frames, worst 3.0 m off route, against 0-1% in the
        three dry episodes that preceded it.

        Grip, not comfort, is binding here. Lateral acceleration is limited by
        the friction circle, so the wet factor is applied to the acceleration
        directly; because v^2 = a*R, a 0.7x acceleration limit is a 0.84x
        cornering speed, which is close to Table 1's own 0.80x wet speed rule.
        """
        limit = self.margin * LAT_ACCEL_MAX_COMFORTABLE
        ctx = self._context_cache
        if ctx is not None and ctx.wet:
            limit *= WET_GRIP_FACTOR
        return limit

    def _plan_lookahead(self, speed: float, cap: float) -> float:
        """How far ahead to *plan*, versus the fixed distance rules are judged on.

        Enough road to decelerate from the current speed to ``cap`` at the
        defensive braking rate, plus the ramp the jerk row forces, plus the
        judging horizon itself so the vehicle is already at the cap when it
        enters the judged zone, plus a reaction allowance. See D25.
        """
        excess = max(speed - cap, 0.0)
        decel = self.margin * BRAKE_DEFENSIVE_TARGET
        braking_distance = excess * excess / (2.0 * decel)
        # The jerk row dominates the distance, not the acceleration row. Torque
        # cannot appear instantly: reaching -decel from zero at the acceptable
        # jerk takes decel/jerk seconds, and the vehicle covers ground at close
        # to its original speed throughout. Omitting this term is what left the
        # expert commanding a stop it had no room to execute.
        jerk_ramp_time = decel / max(self.margin * JERK_LON_ACCEPTABLE, 1e-3)
        jerk_distance = speed * jerk_ramp_time
        return (
            CROSSWALK_JUDGE_LOOKAHEAD
            + braking_distance
            + jerk_distance
            + speed * REACTION_TIME
        )

    def _approaching_crosswalk(
        self, lookahead: float = CROSSWALK_JUDGE_LOOKAHEAD
    ) -> bool:
        """True only for a *marked* crossing.

        D27: this used to return True for any junction within the lookahead.
        Town01 has a junction roughly every 50 m, so once the lookahead grew to
        the distance the jerk limit actually requires, the pedestrian-crossing
        cap of 10 km/h applied almost everywhere and the expert crawled the
        whole episode. Junctions are handled separately, with their own higher
        cap, by ``_approaching_junction``.
        """
        # The test is "is there a crossing ahead *on our path*", not "is there
        # a crossing anywhere nearby". ``get_crosswalks`` returns the polygon
        # vertices of every crossing in the map, so a plain radius test fires
        # on crossings behind the vehicle and on intersecting streets. In
        # Town10HD that meant the 10 km/h cap applied in 76.5% of frames and
        # the ego crawled the whole episode at 6.1 km/h.
        tf = self.vehicle.get_transform()
        location = tf.location
        forward = tf.get_forward_vector()
        for point in self.map.get_crosswalks():
            dx = point.x - location.x
            dy = point.y - location.y
            ahead = dx * forward.x + dy * forward.y
            if ahead <= 0.0 or ahead > lookahead:
                continue
            lateral = abs(-dx * forward.y + dy * forward.x)
            if lateral <= CROSSWALK_LATERAL_TOLERANCE:
                return True
        return False

    def _approaching_junction(self, lookahead: float = 20.0) -> bool:
        wp = self.map.get_waypoint(self.vehicle.get_transform().location)
        if wp.is_junction:
            return True
        return any(a.is_junction for a in (wp.next(lookahead) or []))

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def run_step(self):
        """One control tick. Returns ``(carla.VehicleControl, debug dict)``."""
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        ctx = self.build_context()
        self._context_cache = ctx

        measured_accel = self._filtered_accel()[0]
        target_speed, reason = self._target_speed(ctx, speed)

        # Rate-limit the setpoint itself. ``_target_speed`` takes the minimum
        # over independent ceilings, so the binding constraint can change
        # abruptly (a crosswalk comes into range, a lead vehicle appears) and
        # step the target by several m/s. Feeding that step straight into the
        # loop produces exactly the jerk transient Table 1 forbids.
        if not self.state.filt_initialised:
            self.state.smoothed_target = speed
        max_target_delta = self.margin * LON_ACCEL_MAX_COMFORTABLE * self.dt
        step = target_speed - self.state.smoothed_target
        if abs(step) > max_target_delta:
            # Falling setpoints may move faster than rising ones: slowing down
            # for a hazard must not be rate-limited into a late stop.
            allowance = max_target_delta * (4.0 if step < 0 else 1.0)
            step = math.copysign(min(abs(step), allowance), step)
        self.state.smoothed_target += step

        # Longitudinal: convert the speed error into an acceleration, then clamp
        # it to the Table 1 comfort envelope and rate-limit it for jerk.
        # D49: the speed loop must respect the jerk budget it will have to
        # unwind. A plain proportional term commands an acceleration without
        # asking whether it can be brought back to zero in time, and since D46
        # halved the planned jerk budget, cancelling +1.25 m/s^2 takes over four
        # seconds -- so the ego sailed 1.52 m/s past its own target and sat
        # there, scoring 11.5% of frames as speeding while the binding
        # constraint read "speed_limit". It was not driving too fast, it was
        # unable to stop accelerating.
        #
        # Ramping an acceleration a down to zero at jerk j costs a^2 / 2j of
        # speed, so the largest acceleration that still lands on the target is
        # sqrt(2 * j * error). Taking the tighter of that and the proportional
        # term removes the overshoot by construction rather than by tuning.
        # The cap is applied only when speeding *up*. Overshooting a speed
        # target downwards means arriving slower than necessary, which costs
        # nothing and violates nothing; overshooting upwards is the speeding
        # row. Applying it symmetrically also throttled deceleration into
        # curves, and the ego understeered into a static object -- the cure
        # reintroducing the disease D43 was written to fix.
        error = self.state.smoothed_target - speed
        proportional = error / SPEED_LOOP_TAU

        # D61: when something is inside the standstill gap, demand the
        # deceleration that actually stops short of it rather than whatever the
        # proportional term happens to produce. With SPEED_LOOP_TAU = 1.5, a
        # target of 0 at 1.8 m/s asks for -1.2 m/s^2, while stopping in the
        # remaining 1.31 m needs v^2/2d = 1.24 -- marginal by construction, and
        # it lost by 0.2 m against a stationary bus. The proportional term is a
        # speed-tracking law; it knows nothing about the distance left.
        if self.state.stop_room is not None and speed > 0.05:
            room = max(self.state.stop_room, 0.05)
            stopping = -(speed * speed) / (2.0 * room)
            proportional = min(proportional, stopping)
        if error > 0.0:
            jerk_budget = JERK_PLAN_MARGIN * JERK_LON_ACCEPTABLE
            arrestable = math.sqrt(2.0 * jerk_budget * error)
            raw_accel = min(proportional, arrestable)
        else:
            raw_accel = proportional
        accel = self._limit_longitudinal(raw_accel, target_speed, speed)

        throttle, brake = self._accel_to_pedals(accel, speed, measured_accel)
        steer = self._lateral_control(speed)

        self.state.target_speed = target_speed
        self.state.commanded_accel = accel
        self.state.prev_speed = speed
        self.state.prev_steer = steer

        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steer, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
        )
        debug = {
            "target_speed": target_speed,
            "constraint": reason,
            "accel_cmd": accel,
            "gear": self.vehicle.get_control().gear,
            "crosstrack": self._signed_crosstrack(),
            "ped_distance": self._nearest_pedestrian(),
            "steer": steer,
            "in_junction": self.map.get_waypoint(
                self.vehicle.get_transform().location
            ).is_junction,
        }
        return control, debug

    def _limit_longitudinal(
        self, raw_accel: float, target_speed: float, speed: float
    ) -> float:
        """Clamp to the acceleration rows, then to the jerk row."""
        if raw_accel >= 0:
            accel = min(raw_accel, self.margin * LON_ACCEL_MAX_COMFORTABLE)
        else:
            # Braking picks the gentlest rate that still achieves the target.
            # The emergency rate is available but only as a last resort, and
            # even then stays under the Table 1 ceiling.
            magnitude = abs(raw_accel)
            if magnitude <= BRAKE_NORMAL_TARGET:
                limit = BRAKE_NORMAL_TARGET
            elif magnitude <= BRAKE_DEFENSIVE_TARGET:
                limit = BRAKE_DEFENSIVE_TARGET
            else:
                limit = self.margin * BRAKE_MAX_EMERGENCY
            accel = -min(magnitude, self.margin * limit)

        # Jerk limiting: bound the change in commanded acceleration per tick.
        #
        # D33: this must not apply to an emergency. The comfort jerk row allows
        # the commanded acceleration to change by only 0.51 m/s^2 per second,
        # so reaching the -3.0 m/s^2 that arms the emergency pedal rate took
        # seven seconds -- and the emergency path could therefore never arm
        # itself in time to be useful. Measured consequence: with a stopped
        # vehicle 7 m ahead and a commanded target speed of 0, the expert held
        # 0.36 throttle and zero brake all the way into the impact at 7.6 m/s.
        #
        # Comfort yields to not colliding. The braking demand here is the
        # *unlimited* one, so the escape hatch triggers on the situation rather
        # than on the rate-limited command that the limiter itself produced.
        # The escalation is graded rather than a cliff. A binary trigger at the
        # defensive braking rate meant the budget jumped from 0.51 to 12 m/s^3
        # the instant the demand crossed it, so ordinary firm braking was
        # scored as violently as a genuine emergency. Urgency is measured
        # against the *normal* braking rate and squared, so mild demands stay
        # near the comfort row and only a real hazard spends the full budget.
        urgency = max(1.0, -raw_accel / BRAKE_NORMAL_TARGET) if raw_accel < 0 else 1.0
        jerk_budget = min(
            JERK_PLAN_MARGIN * JERK_LON_ACCEPTABLE * urgency * urgency, JERK_EMERGENCY
        )
        if self.state.proximity_urgent:
            # D60: the reference has to be *allowed* to become a braking
            # command. Rate-limiting it at the comfort budget while an obstacle
            # sits inside the standstill gap means the pedal loop is faithfully
            # tracking a reference that is still asking to creep forward.
            jerk_budget = JERK_EMERGENCY
        max_delta = jerk_budget * self.dt
        delta = accel - self.state.commanded_accel
        if abs(delta) > max_delta:
            accel = self.state.commanded_accel + math.copysign(max_delta, delta)
        return accel

    def _filtered_accel(self) -> Tuple[float, float]:
        """Low-pass filtered (longitudinal, lateral) acceleration.

        D24: CARLA's ``get_acceleration()`` is a one-step difference of
        velocity, so at dt=0.05 it carries the physics solver's per-substep
        noise. Feeding that raw into a controller makes the controller chase
        noise; measured against Table 1 it reports the noise floor rather than
        ride comfort. Both uses take the filtered signal.
        """
        tf = self.vehicle.get_transform()
        a = self.vehicle.get_acceleration()
        yaw = math.radians(tf.rotation.yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        lon = a.x * cos_y + a.y * sin_y
        lat = -a.x * sin_y + a.y * cos_y

        s = self.state
        if not s.filt_initialised:
            s.filt_lon_accel, s.filt_lat_accel = lon, lat
            s.comfort_lon_accel, s.comfort_lat_accel = lon, lat
            s.filt_initialised = True
        else:
            # Control bandwidth: fast enough for the loop to stay stable.
            alpha = self.dt / (ACCEL_FILTER_TAU + self.dt)
            s.filt_lon_accel += alpha * (lon - s.filt_lon_accel)
            s.filt_lat_accel += alpha * (lat - s.filt_lat_accel)

        # Comfort bandwidth, reported to the evaluator. Two stages:
        #  1. a short median, which removes the one-to-two-sample torque steps
        #     an automatic gearbox produces (measured: |jerk| up to 28 m/s^3 at
        #     a shift versus 9 elsewhere) without smearing genuine ramps;
        #  2. a slower one-pole at roughly the bandwidth over which a passenger
        #     integrates motion into a sensation of harshness.
        for window, value in (
            (s.raw_lon_window, lon), (s.raw_lat_window, lat)
        ):
            window.append(value)
            if len(window) > MEDIAN_WINDOW:
                window.pop(0)
        beta = self.dt / (COMFORT_FILTER_TAU + self.dt)
        s.comfort_lon_accel += beta * (_median(s.raw_lon_window) - s.comfort_lon_accel)
        s.comfort_lat_accel += beta * (_median(s.raw_lat_window) - s.comfort_lat_accel)
        return s.filt_lon_accel, s.filt_lat_accel

    def _accel_to_pedals(
        self, accel: float, speed: float, measured_accel: float
    ) -> Tuple[float, float]:
        """Track ``accel`` in closed loop on the *achieved* acceleration.

        D23: the previous version was open loop -- it mapped the commanded
        acceleration to a pedal position with a fixed gain and never checked
        what the vehicle actually did. The command was jerk-limited to 0.026
        m/s^2 per tick while the measured acceleration swung over +-7 m/s^2,
        because CARLA's drag and engine braking dominate the pedal map and
        nothing corrected the residual. Table 1 scores the achieved
        acceleration, so that is what has to be controlled.

        A velocity-form (incremental) PI is used: it carries no separate
        integrator to wind up, and its output moves smoothly by construction,
        which matters when the output *is* the torque the jerk row measures.
        """
        error = accel - measured_accel
        delta = (
            PEDAL_KP * (error - self.state.prev_accel_error)
            + PEDAL_KI * error * self.dt
        )
        self.state.prev_accel_error = error

        # Bound how fast the pedal may move. A pedal step is a torque step, and
        # a torque step is unbounded jerk regardless of what the PI intended.
        # A hard braking demand lifts the cap: exceeding the comfort jerk row
        # for a few frames costs a little compliance, failing to stop costs a
        # collision, and the episode is scored on a rate rather than pass/fail.
        # Graded like the commanded-acceleration budget in ``_limit_longitudinal``,
        # and for the same reason: a binary switch makes ordinary firm braking as
        # abrupt as an emergency. Cruising gets the smooth rate, which is where
        # most frames are and therefore where the jerk row is won or lost.
        pedal_urgency = (
            max(1.0, -accel / BRAKE_NORMAL_TARGET) if accel < 0 else 1.0
        )
        rate = min(
            PEDAL_MAX_RATE * pedal_urgency * pedal_urgency, PEDAL_EMERGENCY_RATE
        )
        # D60: gate on proximity alone. Requiring ``accel <= 0`` here defeated
        # the whole escalation: the commanded acceleration is itself jerk-
        # limited, so while the ego crept the last half-metre into a stationary
        # bus the command was still marginally positive and the override never
        # armed. The repeat run collided at the identical frame.
        if self.state.proximity_urgent:
            rate = PEDAL_EMERGENCY_RATE
        delta = float(np.clip(delta, -rate * self.dt, rate * self.dt))
        pedal = float(np.clip(self.state.pedal + delta, -1.0, 1.0))

        # Hold the vehicle still rather than creeping when fully stopped.
        if speed < 0.1 and accel <= 0.0:
            pedal = min(pedal, -0.3)

        self.state.pedal = pedal
        if pedal >= 0.0:
            return pedal, 0.0
        return 0.0, -pedal

    @staticmethod
    def steer_law(
        heading_error: float,
        crosstrack: float,
        speed: float,
        crosstrack_integral: float,
        k_heading: float,
        k_crosstrack: float,
        k_crosstrack_integral: float,
    ) -> float:
        """The steering law, as a pure function so its signs can be tested.

        ``crosstrack`` is positive when the vehicle is to the *right* of the
        lane centre, and positive ``steer`` turns right, so the crosstrack
        terms are subtracted. See D29 -- getting this backwards is stable but
        leaves a permanent lateral offset.
        """
        return (
            k_heading * heading_error
            - k_crosstrack * math.atan2(crosstrack, max(speed, 1.0))
            - k_crosstrack_integral * crosstrack_integral
        )

    def _signed_crosstrack(self) -> float:
        """Signed lateral offset from the route, positive to the route's right.

        D40: this used to measure against ``map.get_waypoint(location)``, the
        nearest lane centre. That is correct on open road and wrong inside a
        junction, where CARLA's junction lanes overlap and the nearest one is
        frequently the straight-through lane rather than the turn the ego is
        actually taking. The controller then fought its own heading term and
        drifted 1.2 m wide through right turns, and the ``lane_offset`` metric
        graded the ego against a lane it was never on -- 75 of 145 violations
        in one episode were inside junctions.

        The route is built from lane-centre waypoints off the road graph, so it
        is an objective reference that also happens to be defined continuously
        through a junction.
        """
        location = self.vehicle.get_transform().location
        if len(self.route) < 2:
            return 0.0

        # Search a window around the current cursor rather than the whole
        # route: this runs every tick.
        lo = max(0, self._route_index - ROUTE_SEARCH_BACK)
        hi = min(len(self.route) - 1, self._route_index + ROUTE_SEARCH_FORWARD)

        best = None
        for i in range(lo, hi):
            a, b = self.route[i], self.route[i + 1]
            dx, dy = b.x - a.x, b.y - a.y
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-9:
                continue
            # Project the ego onto the segment, clamped to its extent.
            t = ((location.x - a.x) * dx + (location.y - a.y) * dy) / length_sq
            t = min(max(t, 0.0), 1.0)
            px, py = a.x + t * dx, a.y + t * dy
            gap_sq = (location.x - px) ** 2 + (location.y - py) ** 2
            if best is None or gap_sq < best[0]:
                # Right-hand normal of the segment, matching D29's convention.
                length = math.sqrt(length_sq)
                nx, ny = -dy / length, dx / length
                signed = (location.x - a.x) * nx + (location.y - a.y) * ny
                best = (gap_sq, signed)
        return best[1] if best is not None else 0.0

    def _lateral_control(self, speed: float) -> float:
        """Stanley-style steering with a rate limit for lateral jerk."""
        tf = self.vehicle.get_transform()
        target = self._lookahead_point(speed)
        if target is None:
            return self.state.prev_steer

        yaw = math.radians(tf.rotation.yaw)
        dx = target.x - tf.location.x
        dy = target.y - tf.location.y
        target_yaw = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)
        )

        # Cross-track error against the route the ego intends to follow (D40).
        wp = self.map.get_waypoint(tf.location)
        crosstrack = self._signed_crosstrack()

        # D28: plain Stanley is proportional only, so a constant disturbance
        # leaves a standing offset it cannot remove. Measured over an episode:
        # crosstrack was negative in 99.5% of frames, mean -0.208 m against a
        # 0.30 m limit, and all 110 violations were on open lane and on the
        # same side -- the signature of steady-state error, not of noise or of
        # cornering. A bounded integral term cancels it.
        #
        # The authority is sized, not guessed: it has to cancel a steer bias of
        # order 0.015, so the clamped contribution is 0.025 x 1.0 = 0.025. An
        # earlier attempt used 0.12 x 2.0 = 0.24 -- comparable to full lock --
        # and drove the car 11 m off the road.
        if wp.is_junction:
            # Lane centre is ambiguous through a junction and the geometry
            # changes under the vehicle; holding the accumulator there winds it
            # up against a reference that is about to move.
            self.state.crosstrack_integral *= JUNCTION_INTEGRAL_DECAY
        else:
            self.state.crosstrack_integral = float(
                np.clip(
                    self.state.crosstrack_integral * CROSSTRACK_INTEGRAL_LEAK
                    + crosstrack * self.dt,
                    -CROSSTRACK_INTEGRAL_LIMIT,
                    CROSSTRACK_INTEGRAL_LIMIT,
                )
            )
        # D29: the crosstrack terms are *subtracted*, not added.
        #
        # CARLA's world is left-handed (x forward, y right, yaw clockwise seen
        # from above), so the lane's right-hand normal is (-sin, cos) and
        # ``crosstrack`` as computed above is positive when the vehicle sits to
        # the *right* of the lane centre. Positive ``steer`` also turns right.
        # Adding the term therefore steered further from the centre the further
        # off-centre the vehicle was: positive feedback.
        #
        # It went unnoticed because the heading term has the correct sign and
        # is stronger, so the loop stayed stable and merely settled at a
        # standing offset -- crosstrack negative in 99.5% of frames. Raising the
        # crosstrack gain, or adding an integral, amplified the wrong sign and
        # made the offset dramatically worse (85% of frames violating, and in
        # one attempt 11 m off the road) which is what exposed it.
        steer = self.steer_law(
            heading_error,
            crosstrack,
            speed,
            self.state.crosstrack_integral,
            self.k_heading,
            self.k_crosstrack,
            self.k_crosstrack_integral,
        )

        # D26: cap the steering angle by the lateral acceleration it would
        # produce. a_lat = v^2 / R and R ~ L / tan(delta), so a steer angle that
        # is comfortable at walking pace breaks the lateral acceleration row at
        # 50 km/h. Rate-limiting alone cannot prevent this -- it bounds how fast
        # the wheel turns, not where it ends up.
        # D35: remember what the law *wanted* before the comfort cap, so the
        # speed target can be lowered until the demanded turn becomes feasible.
        # Clipping the angle alone makes the vehicle understeer: it silently
        # fails to follow the route while believing it is complying. Measured:
        # through a junction the steer sat pinned at the 0.109 ceiling for ten
        # consecutive frames while lane offset grew 1.5 -> 3.3 m and the ego
        # ran wide into a static object.
        self.state.desired_steer = steer

        if speed > 1.0:
            max_lat = self._lateral_limit()
            max_curvature = max_lat / (speed * speed)
            max_angle = math.atan(WHEELBASE * max_curvature)
            steer_ceiling = min(1.0, max_angle / MAX_STEER_ANGLE)
            steer = float(np.clip(steer, -steer_ceiling, steer_ceiling))

        # Rate limit -> bounds lateral jerk, which Table 1 caps at 0.42 m/s^3.
        # The bound is itself speed-dependent for the same reason: at speed v a
        # steer rate of d(delta)/dt produces lateral jerk proportional to v^2.
        rate_cap = self.max_steer_rate
        if speed > 1.0:
            jerk_limited = (
                self.margin * JERK_LAT_ACCEPTABLE * WHEELBASE
                / (speed * speed * MAX_STEER_ANGLE)
            )
            rate_cap = min(rate_cap, max(jerk_limited * self.dt, MIN_STEER_RATE))
        # D43: the comfort rate must not trap the vehicle off its path. Once
        # the lateral error is large the ego is no longer merely uncomfortable,
        # it is in the wrong place -- and every frame spent slewing the wheel
        # at the comfort rate takes it further. Same principle as D33.
        if abs(crosstrack) > CROSSTRACK_RECOVERY_THRESHOLD:
            rate_cap = max(rate_cap, STEER_RECOVERY_RATE)
        delta = steer - self.state.prev_steer
        if abs(delta) > rate_cap:
            steer = self.state.prev_steer + math.copysign(rate_cap, delta)
        return steer

    def _lookahead_point(self, speed: float):
        """Point on the route ~1.5 s ahead, advancing the route cursor."""
        if not self.route:
            return None
        # D41: a pure-pursuit follower cuts corners by an amount that grows
        # with the lookahead distance and the path curvature. At a fixed 5 m
        # minimum, junction turns of ~6 m radius were cut badly: measured mean
        # offset 1.159 m while steering hard against 0.161 m running straight,
        # and only 5.7% of straight open-lane frames violated at all. The
        # lookahead now scales down with speed and floors much lower, so tight
        # low-speed turns are tracked rather than short-cut.
        distance = max(LOOKAHEAD_MIN, speed * LOOKAHEAD_TIME)
        location = self.vehicle.get_transform().location

        # D42: advance the cursor by projection, not by proximity.
        #
        # The previous version stepped the cursor forward only while the point
        # it currently pointed at was within 3 m of the ego. The moment the ego
        # got further than that -- which a corner cut alone can do -- the cursor
        # froze permanently, and the ego then chased a lookahead measured from a
        # stale index, eventually one *behind* it, and diverged with no path
        # back. That is why shortening the lookahead turned a 2 m tracking
        # error into a 5 m runaway rather than improving it.
        #
        # Picking the nearest point in a forward window is robust to being far
        # off route, and stays monotonic so the ego cannot be sent backwards.
        best_index, best_distance = self._route_index, None
        horizon = min(len(self.route), self._route_index + ROUTE_SEARCH_FORWARD)
        for i in range(self._route_index, horizon):
            gap = self.route[i].distance(location)
            if best_distance is None or gap < best_distance:
                best_distance, best_index = gap, i
        self._route_index = best_index

        for i in range(self._route_index, len(self.route)):
            if self.route[i].distance(location) >= distance:
                return self.route[i]
        return self.route[-1]

    # ------------------------------------------------------------------
    # Observation, for the compliance evaluator
    # ------------------------------------------------------------------

    def observe(self, prev: Optional[EgoObservation] = None) -> EgoObservation:
        """Build the frame's ``EgoObservation`` from simulator state.

        Jerk needs the previous frame's acceleration, so ``prev`` is required
        to produce non-zero jerk; on the first frame it is legitimately zero.
        """
        tf = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # D24: report the filtered acceleration, not the raw one-step velocity
        # difference. Table 1's acceleration and jerk rows are ride-comfort
        # limits on the sustained motion a passenger feels; the raw signal at
        # dt=0.05 is dominated by the physics solver's per-substep noise, and
        # differentiating it again for jerk amplifies that noise by 20x.
        # ``_filtered_accel`` is already advanced once per tick by ``run_step``,
        # so this reads the current value rather than filtering twice.
        lon_accel = self.state.comfort_lon_accel
        lat_accel = self.state.comfort_lat_accel

        if prev is not None:
            lon_jerk = (lon_accel - prev.lon_accel) / self.dt
            lat_jerk = (lat_accel - prev.lat_accel) / self.dt
        else:
            lon_jerk = 0.0
            lat_jerk = 0.0

        # D40: measured against the route, which is continuous through a
        # junction, rather than the nearest lane centre, which is not.
        lane_offset = abs(self._signed_crosstrack())

        lead, lead_distance = self._find_lead_vehicle()
        lead_speed = None
        if lead is not None:
            lv = lead.get_velocity()
            lead_speed = math.sqrt(lv.x**2 + lv.y**2 + lv.z**2)

        light_state, light_distance = self._traffic_light_state()
        is_red = light_state is not None and "Red" in light_state
        is_yellow = light_state is not None and "Yellow" in light_state

        # D64: a red-light run is the *event* of crossing the stop line while
        # the light is red, not a speed check inside a 1 m band around it.
        # Detected here because it needs the previous frame's signed distance,
        # which ``evaluate_frame`` does not have.
        ran_red = False
        previous = self.state.stop_line_distance
        if (
            is_red
            and light_distance is not None
            and previous is not None
            and previous > 0.0
            and light_distance <= 0.0
        ):
            ran_red = True
        self.state.stop_line_distance = light_distance if is_red else None

        return EgoObservation(
            speed=speed,
            lon_accel=lon_accel,
            lat_accel=lat_accel,
            lon_jerk=lon_jerk,
            lat_jerk=lat_jerk,
            lane_offset=lane_offset,
            lead_distance=lead_distance,
            lead_speed=lead_speed,
            at_red_light_or_stop=is_red,
            ran_red_light=ran_red,
            distance_to_stop_line=light_distance,
            at_yellow_light=is_yellow,
            is_decelerating=self.state.commanded_accel < -0.05,
            nearest_pedestrian_distance=self._nearest_pedestrian(),
            approaching_crosswalk=self._approaching_crosswalk(),
        )
