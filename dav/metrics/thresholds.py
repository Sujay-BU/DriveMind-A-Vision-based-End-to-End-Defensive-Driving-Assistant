"""Defensive driving thresholds, transcribed from Table 1 of the thesis.

Every constant in this module traces back to a row of Table 1.  Nothing here is
tuned or invented; where the thesis is ambiguous the resolution is recorded in
the ``NOTE`` comment beside the constant and repeated in DOCUMENTATION.md.

Units are SI throughout (m, s, m/s, m/s^2, m/s^3) even where Table 1 quotes
km/h or G, so that the conversions live in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

G = 9.80665  # m/s^2
KMH_TO_MS = 1.0 / 3.6


# --------------------------------------------------------------------------
# Context enums.  The thresholds in Table 1 are conditional on the driving
# context, so the context has to be a first-class object rather than a set of
# loose booleans.
# --------------------------------------------------------------------------


class RoadClass(Enum):
    """Road classes used by the following-distance rows of Table 1.

    NOTE: Table 1 lists "city" (3.0 s) and "urban" (4.0 s) as distinct rows.
    In ordinary usage these are synonyms, and the thesis never defines them.
    Resolved by speed band, which is the only reading under which the ordering
    3.0 < 4.0 < 5.0 makes physical sense (longer gaps at higher speeds):
        CITY    - dense street grid, posted limit <= 40 km/h
        URBAN   - arterial / suburban road, 40 < posted limit <= 70 km/h
        HIGHWAY - freeway, posted limit > 70 km/h
    """

    CITY = "city"
    URBAN = "urban"
    HIGHWAY = "highway"

    @staticmethod
    def from_speed_limit(limit_ms: float) -> "RoadClass":
        limit_kmh = limit_ms / KMH_TO_MS
        if limit_kmh <= 40.0:
            return RoadClass.CITY
        if limit_kmh <= 70.0:
            return RoadClass.URBAN
        return RoadClass.HIGHWAY


@dataclass(frozen=True)
class DrivingContext:
    """Everything Table 1's conditional thresholds depend on."""

    speed_limit_ms: float
    wet: bool = False
    low_visibility: bool = False
    night: bool = False
    lead_is_heavy_vehicle: bool = False
    road_class: RoadClass | None = None

    def resolved_road_class(self) -> RoadClass:
        return self.road_class or RoadClass.from_speed_limit(self.speed_limit_ms)

    @property
    def adverse_weather(self) -> bool:
        """Table 1's "adverse weather" multiplier trigger.

        NOTE: Table 1 gives a x1.5 following-distance multiplier for "adverse
        weather" without defining the term. Taken to mean wet road or reduced
        visibility, i.e. exactly the conditions that already trigger a speed
        reduction elsewhere in the same table.
        """
        return self.wet or self.low_visibility


# --------------------------------------------------------------------------
# Row 1 - Speed.  All values are multipliers on the posted speed limit.
# --------------------------------------------------------------------------

SPEED_FACTOR_MAX_OPERATING = 0.95
SPEED_FACTOR_WET = 0.80  # "-20% of the posted speed limit"
SPEED_FACTOR_LOW_VISIBILITY = 0.70  # "-30%"
SPEED_FACTOR_NIGHT = 0.90  # "-10%"
SPEED_FACTOR_COMBINED = 0.75  # "-25%", wet + low visibility + night


def speed_limit_for(ctx: DrivingContext) -> float:
    """Defensive speed ceiling in m/s for a given context.

    The combined row supersedes the individual rows when all three conditions
    hold; otherwise the most restrictive applicable factor wins.  The maximum
    operating factor (0.95) is always in force as an upper bound.
    """
    if ctx.wet and ctx.low_visibility and ctx.night:
        factor = SPEED_FACTOR_COMBINED
    else:
        factor = SPEED_FACTOR_MAX_OPERATING
        if ctx.wet:
            factor = min(factor, SPEED_FACTOR_WET)
        if ctx.low_visibility:
            factor = min(factor, SPEED_FACTOR_LOW_VISIBILITY)
        if ctx.night:
            factor = min(factor, SPEED_FACTOR_NIGHT)
    return factor * ctx.speed_limit_ms


# --------------------------------------------------------------------------
# Row 2 - Following distance, expressed as a time gap in seconds.
# --------------------------------------------------------------------------

FOLLOWING_GAP_S: Dict[RoadClass, float] = {
    RoadClass.CITY: 3.0,
    RoadClass.URBAN: 4.0,
    RoadClass.HIGHWAY: 5.0,
}
FOLLOWING_MULT_ADVERSE_WEATHER = 1.5
FOLLOWING_MULT_HEAVY_VEHICLE = 1.3

# Saturation cap for the measured time gap.
# NOTE: not from Table 1. The time gap is undefined when there is no lead
# vehicle (it diverges to infinity), which cannot be fed to a network or a
# loss. Frames with no lead vehicle are masked out of the loss entirely; this
# cap only bounds the value written to disk so the dataset has no infinities.
FOLLOWING_GAP_CAP_S = 10.0


def following_gap_for(ctx: DrivingContext) -> float:
    """Required time gap to the lead vehicle in seconds."""
    gap = FOLLOWING_GAP_S[ctx.resolved_road_class()]
    if ctx.adverse_weather:
        gap *= FOLLOWING_MULT_ADVERSE_WEATHER
    if ctx.lead_is_heavy_vehicle:
        gap *= FOLLOWING_MULT_HEAVY_VEHICLE
    return gap


# --------------------------------------------------------------------------
# Row 3 - Longitudinal acceleration.
# --------------------------------------------------------------------------

LON_ACCEL_MAX_COMFORTABLE = 0.15 * G  # 1.47 m/s^2
LON_ACCEL_HARSH = 0.30 * G  # 2.94 m/s^2
BRAKE_MAX_EMERGENCY = 0.47 * G  # 4.61 m/s^2
BRAKE_DEFENSIVE_TARGET = 0.30 * G  # 2.94 m/s^2
BRAKE_NORMAL_TARGET = 0.15 * G  # 1.47 m/s^2

# --------------------------------------------------------------------------
# Row 4 - Lateral acceleration.
# --------------------------------------------------------------------------

LAT_ACCEL_MAX_COMFORTABLE = 0.20 * G  # 1.96 m/s^2
LAT_ACCEL_HARSH = 0.47 * G  # 4.61 m/s^2
LAT_ACCEL_DEFENSIVE = 0.20 * G  # 1.96 m/s^2

# --------------------------------------------------------------------------
# Row 5 - Jerk.
# --------------------------------------------------------------------------

JERK_LON_ACCEPTABLE = 0.60  # m/s^3
JERK_LAT_ACCEPTABLE = 0.42  # m/s^3
JERK_HARD_LIMIT = 2.94  # m/s^3

# --------------------------------------------------------------------------
# Row 6 - Traffic rules.
# --------------------------------------------------------------------------

YIELD_SPEED_RED_LIGHT = 0.5 * KMH_TO_MS  # < 0.5 km/h at a red light / stop sign
YELLOW_LIGHT_DECEL_DISTANCE = 30.0  # m; begin decelerating if further than this
LANE_OFFSET_MAX = 0.3  # m; ride-quality target for lane centring (diagnostic
#: only -- see D48, this is not a Table 1 row and does not gate compliance)
#: Lateral deviation at which a wheel crosses the lane marking, m. A CARLA lane
#: is 3.5 m and the vehicle ~1.8 m wide, so the centre may deviate this far
#: before any part of the vehicle leaves the lane. *This* is the traffic rule.
LANE_DEPARTURE_MAX = 0.85
#: Distance before a red stop line within which the ego must be stationary, m.
#:
#: Narrow on purpose. It was widened to 4.0 m in D64 so that running a light
#: would register, but with the crossing itself now caught as ``red_light_run``
#: (D66) that is unnecessary -- and a wide band scores the *approach* as a
#: violation, because the vehicle has to decelerate through it. 5.4% of frames
#: were flagged for the entirely correct behaviour of braking to a halt.
#: "Failed to stop at the line" and "drove through the light" are separate
#: rules; this one is only the former.
RED_LIGHT_STOP_BAND = 1.5
PEDESTRIAN_YIELD_DISTANCE = 5.0  # m clearance to a pedestrian
CROSSWALK_APPROACH_SPEED = 10.0 * KMH_TO_MS  # max speed approaching a crosswalk


# --------------------------------------------------------------------------
# The six scalars the safety query is trained against.
#
# The thesis names these in Chapter 3: "longitudinal acceleration, lateral
# acceleration, longitudinal jerk, lateral jerk, city following distance and
# vehicle maximum operating speed".
#
# Five of the six are "smaller is safer". Following distance is the exception:
# a larger gap is safer. To give the asymmetric loss a single consistent
# direction, every scalar is converted to a *violation ratio* where 1.0 means
# exactly at threshold and > 1.0 means unsafe. For following distance that
# means inverting the ratio (required / measured) rather than (measured /
# required). See DOCUMENTATION.md, deviation D12.
# --------------------------------------------------------------------------

SAFETY_METRIC_NAMES = (
    "lon_accel",
    "lat_accel",
    "lon_jerk",
    "lat_jerk",
    "following_gap",
    "speed",
)
NUM_SAFETY_METRICS = len(SAFETY_METRIC_NAMES)


@dataclass(frozen=True)
class SafetyMetricSpec:
    """How one of the six loss metrics maps to a violation ratio."""

    name: str
    #: Threshold in SI units. ``None`` means context-dependent, supplied at runtime.
    threshold: float | None
    #: True when a *larger* measured value is safer (following distance only).
    higher_is_safer: bool = False
    #: Weight w_i in the safety Huber sum.
    weight: float = 1.0


SAFETY_METRIC_SPECS: Dict[str, SafetyMetricSpec] = {
    "lon_accel": SafetyMetricSpec("lon_accel", LON_ACCEL_MAX_COMFORTABLE),
    "lat_accel": SafetyMetricSpec("lat_accel", LAT_ACCEL_MAX_COMFORTABLE),
    "lon_jerk": SafetyMetricSpec("lon_jerk", JERK_LON_ACCEPTABLE),
    "lat_jerk": SafetyMetricSpec("lat_jerk", JERK_LAT_ACCEPTABLE),
    "following_gap": SafetyMetricSpec(
        "following_gap", None, higher_is_safer=True, weight=1.5
    ),
    "speed": SafetyMetricSpec("speed", None),
}

#: Order used for every fixed-width safety vector on disk and in the model.
SAFETY_VECTOR_ORDER = SAFETY_METRIC_NAMES


def safety_thresholds_for(ctx: DrivingContext) -> Dict[str, float]:
    """Resolve all six thresholds for a context, including the two dynamic ones."""
    return {
        "lon_accel": LON_ACCEL_MAX_COMFORTABLE,
        "lat_accel": LAT_ACCEL_MAX_COMFORTABLE,
        "lon_jerk": JERK_LON_ACCEPTABLE,
        "lat_jerk": JERK_LAT_ACCEPTABLE,
        "following_gap": following_gap_for(ctx),
        "speed": speed_limit_for(ctx),
    }


# --------------------------------------------------------------------------
# Weather presets.
#
# NOTE: the thesis says "11 weather conditions" and then lists ten:
# clear noon, clear sunset, cloudy noon, wet noon, wet cloudy noon,
# mid rain noon, hard rain noon, clear night, wet night, hard rain night.
# SoftRainNoon is added to reach the stated count of 11; it is the one CARLA
# preset that fills the obvious gap in the listed rain progression
# (soft -> mid -> hard). See DOCUMENTATION.md, deviation D9.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherPreset:
    """A CARLA weather preset tagged with the Table 1 conditions it triggers."""

    carla_name: str
    wet: bool
    low_visibility: bool
    night: bool


WEATHER_PRESETS: tuple[WeatherPreset, ...] = (
    WeatherPreset("ClearNoon", False, False, False),
    WeatherPreset("ClearSunset", False, False, False),
    WeatherPreset("CloudyNoon", False, False, False),
    WeatherPreset("WetNoon", True, False, False),
    WeatherPreset("WetCloudyNoon", True, False, False),
    WeatherPreset("SoftRainNoon", True, False, False),  # added; see note above
    WeatherPreset("MidRainyNoon", True, True, False),
    WeatherPreset("HardRainNoon", True, True, False),
    WeatherPreset("ClearNight", False, False, True),
    WeatherPreset("WetNight", True, False, True),
    WeatherPreset("HardRainNight", True, True, True),
)

assert len(WEATHER_PRESETS) == 11, "thesis specifies 11 weather conditions"


#: The eight towns of the thesis. CARLA 0.9.15 ships exactly these eight
#: without an extra asset download (Town06/07 and Town10HD are in the release).
TOWNS: tuple[str, ...] = (
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
)


#: Vehicle blueprint substrings treated as heavy vehicles for the x1.3 row.
HEAVY_VEHICLE_KEYWORDS: tuple[str, ...] = (
    "truck",
    "carlacola",
    "cybertruck",
    "firetruck",
    "ambulance",
    "sprinter",
    "bus",
    "van",
)


def is_heavy_vehicle(blueprint_id: str) -> bool:
    lowered = blueprint_id.lower()
    return any(k in lowered for k in HEAVY_VEHICLE_KEYWORDS)
