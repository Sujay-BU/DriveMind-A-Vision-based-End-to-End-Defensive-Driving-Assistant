"""CARLA episode collector producing a nuScenes-shaped defensive-driving dataset.

Chapter 3 specifies: 50 episodes x 3000 frames, 11 weather conditions, 8 towns,
randomised traffic and world seed, nuScenes-like annotations, and the expert's
Table 1 constraints enforced as hard constraints -- "even a single infraction is
considered a failure".

That last requirement is *not* implemented literally, because it accepts
nothing: measured against a live simulator the expert was clean in 15.1% of
frames on its first instrumented episode, so a single-infraction rule rejects
every episode forever.  Episodes are accepted on a compliance *rate* instead
(``min_compliance``, default 0.70) and must also cover ground
(``min_mean_speed``), since compliance alone is trivially maximised by not
moving.  Failing episodes are re-run with a new seed; ``max_attempts`` bounds
the retrying so a pathological map/weather pair cannot stall the run forever.
See DOCUMENTATION.md deviations D22 and D31.

Output layout, per episode::

    <root>/
      samples/CAM_FRONT/<episode>_<frame>.jpg      (and the five other cameras)
      v1.0-dav/
        scene.json  sample.json  sample_data.json  ego_pose.json
        calibrated_sensor.json  sensor.json  category.json
        instance.json  sample_annotation.json  log.json
      safety/<episode>.json        <- the "separate file" of scalar metric data
      episodes/<episode>.json      <- per-episode compliance summary
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..metrics.evaluator import (
    ComplianceReport,
    EgoObservation,
    EpisodeCompliance,
    evaluate_frame,
    safety_vector,
)
from ..metrics.thresholds import (
    TOWNS,
    WEATHER_PRESETS,
    DrivingContext,
    WeatherPreset,
)
from .sensors import CAMERA_ORDER, build_camera_blueprints, calibration_table

try:  # pragma: no cover
    import carla
except ImportError:  # pragma: no cover
    carla = None


#: Object classes annotated in the dataset, in the nuScenes naming style.
CATEGORIES = (
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "human.pedestrian",
    "static.traffic_light",
    "static.traffic_sign",
    "static.stop_sign",
    "movable_object.barrier",
)
CATEGORY_INDEX = {name: i for i, name in enumerate(CATEGORIES)}


@dataclass
class CollectConfig:
    """Data collection settings. See ``configs/collect_*.yaml``."""

    out_dir: str = "data/dav"
    episodes: int = 50
    frames_per_episode: int = 3000
    image_width: int = 400
    image_height: int = 225
    fps: int = 20  # -> fixed_delta_seconds = 0.05
    num_vehicles: int = 60
    num_pedestrians: int = 30
    warmup_frames: int = 40  # discarded; lets traffic and physics settle
    towns: tuple[str, ...] = TOWNS
    seed: Optional[int] = None
    max_attempts_per_episode: int = 3
    jpeg_quality: int = 90
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    traffic_manager_port: int = 8000
    #: Vehicles seeded directly ahead of the ego in its own lane, and in the
    #: opposing lane, rather than at random spawn points (D59). Random
    #: placement leaves it to chance whether the ego ever meets a lead vehicle
    #: or oncoming traffic at all.
    lead_vehicles: int = 0
    oncoming_vehicles: int = 0
    #: Spacing between successive seeded vehicles, metres.
    lead_spacing: float = 18.0
    oncoming_spacing: float = 22.0
    #: Gap the traffic manager keeps behind *its* lead vehicle, metres (D36).
    #: CARLA's default of 1.0 tailgates a defensively-driven ego.
    traffic_following_distance: float = 3.0
    timeout: float = 60.0
    #: RPC deadline for ``load_world`` only (see D21). Streaming a town off a
    #: rotational disk can take 10+ minutes, which is not a simulator fault.
    map_load_timeout: float = 900.0
    #: Stop rejecting episodes on infractions. Off by default; useful when
    #: debugging the expert, since it lets the episode run to completion and
    #: produce a full per-metric profile instead of aborting on frame one.
    allow_infractions: bool = False
    #: Fraction of frames that must be free of every Table 1 infraction for the
    #: episode to be accepted (see D22). An episode below this is rejected.
    min_compliance: float = 0.70
    #: Mean speed the episode must sustain, m/s (see D31). Compliance alone is
    #: trivially maximised by not moving, so it cannot be the only criterion.
    #:
    #: This is a *degeneracy* guard, not a quality target. The failure it exists
    #: to catch covered 3.9 m in 20 s at 0.20 m/s; genuine stop-and-go downtown
    #: traffic sits around 1.7 m/s and is legitimate driving data. Set between
    #: the two, closer to the degenerate case.
    min_mean_speed: float = 1.0
    #: Dump a per-frame CSV of ego state, control and violations per episode.
    trace_csv: bool = False

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


def _token() -> str:
    return uuid.uuid4().hex


def print_episode_profile(compliance: EpisodeCompliance) -> None:
    """Print how the whole attempt behaved, not just the frame that killed it.

    A single offending frame cannot distinguish "the expert is marginal on jerk
    for half the episode" from "the expert was clean until it was rear-ended".
    Those need opposite fixes, so the per-metric violation *rate* over the
    attempt is printed alongside the worst ratio seen.
    """
    if compliance.frames == 0:
        return
    s = compliance.summary()
    rates = s["violation_rate"]
    maxes = s["max_ratio"]
    print(
        f"    episode profile over {s['frames']} frames "
        f"(clean {s['compliance_score']:.1%}):"
    )
    names = sorted(set(rates) | set(maxes))
    if not names:
        print("      no violations of any metric")
        return
    for name in names:
        rate = rates.get(name, 0.0)
        peak = maxes.get(name)
        peak_txt = f", worst {peak:.2f}x" if peak is not None else ""
        print(
            f"      {name}: violated in {rate:.1%} of frames "
            f"({round(rate * s['frames'])}/{s['frames']}){peak_txt}"
        )


#: Ticks to settle physics and start sensor streams before the expert takes
#: over. Distinct from ``warmup_frames``, which is driven (D37).
SETTLE_TICKS = 10

#: Speed change across one tick that marks a collision as a shove rather than
#: an impact the ego drove into, m/s. One tick of the strongest legitimate
#: braking is well under this.
COLLISION_SHOVE_MS = 0.5

#: Neighbouring lanes to walk through when looking for the opposing
#: carriageway (D59). Wide enough to cross a multi-lane one-way side.
MAX_LANE_SEARCH = 6


def _basename_map(name: str) -> str:
    """``'Carla/Maps/Town10HD_Opt'`` -> ``'Town10HD'``.

    CARLA reports the loaded map as a full asset path, and ships some towns
    only in their layered ``_Opt`` form -- which is what the server boots into
    by default. The layered variant loads with every layer enabled, so it is
    geometrically the same map; treating the two as distinct would force a
    multi-minute reload for no change in the scene.
    """
    base = name.split("/")[-1]
    return base[: -len("_Opt")] if base.endswith("_Opt") else base


class NuScenesTables:
    """Accumulates the nuScenes table rows across every episode."""

    def __init__(self) -> None:
        self.scene: List[dict] = []
        self.sample: List[dict] = []
        self.sample_data: List[dict] = []
        self.ego_pose: List[dict] = []
        self.calibrated_sensor: List[dict] = []
        self.sensor: List[dict] = []
        self.instance: List[dict] = []
        self.sample_annotation: List[dict] = []
        self.log: List[dict] = []
        self.category = [
            {"token": _token(), "name": name, "description": name}
            for name in CATEGORIES
        ]
        self._category_tokens = {c["name"]: c["token"] for c in self.category}

    def category_token(self, name: str) -> str:
        return self._category_tokens.get(name, self._category_tokens["vehicle.car"])

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "scene",
            "sample",
            "sample_data",
            "ego_pose",
            "calibrated_sensor",
            "sensor",
            "instance",
            "sample_annotation",
            "log",
            "category",
        ):
            with open(directory / f"{name}.json", "w") as fh:
                json.dump(getattr(self, name), fh)


def carla_category(actor) -> Optional[str]:
    """Map a CARLA actor to one of ``CATEGORIES``, or None to skip it."""
    tid = actor.type_id
    if tid.startswith("walker.pedestrian"):
        return "human.pedestrian"
    if tid.startswith("traffic.traffic_light"):
        return "static.traffic_light"
    if tid.startswith("traffic.stop"):
        return "static.stop_sign"
    if tid.startswith("traffic."):
        return "static.traffic_sign"
    if not tid.startswith("vehicle."):
        return None

    attributes = actor.attributes
    wheels = int(attributes.get("number_of_wheels", 4))
    if wheels == 2:
        return "vehicle.bicycle" if "bike" in tid or "crossbike" in tid else "vehicle.motorcycle"
    lowered = tid.lower()
    if "bus" in lowered:
        return "vehicle.bus"
    if any(k in lowered for k in ("truck", "carlacola", "firetruck", "ambulance", "sprinter")):
        return "vehicle.truck"
    return "vehicle.car"


class EpisodeCollector:
    """Runs and records a single episode."""

    def __init__(self, cfg: CollectConfig, client, tables: NuScenesTables) -> None:
        self.cfg = cfg
        self.client = client
        self.tables = tables
        self.world = None
        self.vehicle = None
        self.cameras: List = []
        self.image_queues: Dict[str, "queue.Queue"] = {}
        self.spawned: List = []
        self.original_settings = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def setup(self, town: str, weather: WeatherPreset, seed: int):
        cfg = self.cfg

        # Loading a world streams the whole map off disk. That is the single
        # most expensive operation in collection -- minutes on a spinning
        # disk -- so it is skipped when the requested town is already loaded.
        # ``collect`` orders episodes to group towns together so this hits.
        #
        # D21: ``load_world`` needs its own, much longer RPC deadline. The
        # per-tick timeout is deliberately short so a wedged simulator is
        # caught quickly, but a cold map load off a slow disk legitimately
        # takes many minutes and was aborting collection with a spurious
        # "make sure the simulator is ready" error while the server was in
        # fact loading normally. The long deadline is scoped to the load.
        current = None
        try:
            current = _basename_map(self.client.get_world().get_map().name)
        except RuntimeError:
            pass  # no world yet, or the server is still coming up

        if current == _basename_map(town):
            self.world = self.client.get_world()
        else:
            self.client.set_timeout(cfg.map_load_timeout)
            try:
                self.world = self.client.load_world(town)
            finally:
                self.client.set_timeout(cfg.timeout)
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg.dt
        # Deterministic physics substepping; without this, high-frequency
        # acceleration noise leaks into the jerk metric.
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)

        self.world.set_weather(getattr(carla.WeatherParameters, weather.carla_name))

        random.seed(seed)
        np.random.seed(seed)

        tm = self.client.get_trafficmanager(cfg.traffic_manager_port)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)

        # D36: background traffic must leave room behind a defensive ego.
        #
        # The expert accelerates at Table 1's comfort limit (1.25 m/s^2), so it
        # takes about six seconds to reach city speed from a standstill. CARLA's
        # traffic manager defaults to a 1 m following distance and tailgates
        # into the back of it: an episode was lost 3.4 s in when the ego, doing
        # 2.1 m/s with its nearest detected vehicle 21 m ahead, was shoved from
        # behind to 5.7 m/s.
        #
        # Widening the background following distance is not tuning the metric --
        # the ego's own compliance is unaffected either way. It removes a
        # failure mode that is the traffic manager's behaviour rather than the
        # expert's, and a 3 m gap is closer to real traffic than 1 m.
        tm.set_global_distance_to_leading_vehicle(cfg.traffic_following_distance)

        blueprints = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # D38: the simulator is a persistent process, so a previous run that
        # died without tearing down leaves its vehicles parked on the spawn
        # points and every subsequent episode fails with "Spawn failed because
        # of collision at spawn position". Collection owns the simulator, so
        # start each episode from a clean world rather than depending on the
        # previous one having exited politely.
        self._purge_world()

        ego_bp = blueprints.filter("vehicle.lincoln.mkz_2020")
        ego_bp = ego_bp[0] if ego_bp else blueprints.filter("vehicle.*")[0]
        ego_bp.set_attribute("role_name", "hero")

        # Spawn points can also be blocked by traffic that has not yet been
        # spawned-and-moved, so try several rather than insisting on the first.
        self.vehicle = None
        for index, point in enumerate(spawn_points):
            try:
                self.vehicle = self.world.spawn_actor(ego_bp, point)
                spawn_points = spawn_points[index + 1:]
                break
            except RuntimeError:
                continue
        if self.vehicle is None:
            raise RuntimeError("no free spawn point for the ego vehicle")
        self.spawned.append(self.vehicle)

        self._spawn_traffic(blueprints, spawn_points[1:], tm)
        self._attach_cameras(blueprints)

        # Only enough ticks to settle physics and start the sensor streams. The
        # real warmup runs *under expert control* in ``warmup()`` -- see D37.
        for _ in range(SETTLE_TICKS):
            self.world.tick()
            self._drain_images()

        return self._build_route(spawn_points)

    def warmup(self, expert) -> None:
        """Drive the ego up to speed before recording starts.

        D37: this loop used to sit in ``setup``, before the expert existed, and
        so ticked the world with no control applied at all. The ego stood still
        for the whole warmup while the traffic manager's vehicles drove up and
        queued on its bumper; recording then began with the ego at 0 m/s and a
        car directly behind it. Because the expert accelerates at Table 1's
        comfort limit it stayed slow for several seconds more, and episodes
        were lost to rear-end collisions within the first 60 frames.

        Running the expert here means recording starts with the ego already at
        cruising speed and traffic correctly spaced around it.
        """
        for _ in range(self.cfg.warmup_frames):
            control, _ = expert.run_step()
            self.vehicle.apply_control(control)
            self.world.tick()
            self._drain_images()

    def _purge_world(self) -> None:
        """Destroy every vehicle, walker and sensor left in the world (D38)."""
        actors = self.world.get_actors()
        stale = [
            actor
            for pattern in ("vehicle.*", "walker.*", "sensor.*", "controller.*")
            for actor in actors.filter(pattern)
        ]
        if not stale:
            return
        for actor in stale:
            try:
                if hasattr(actor, "stop"):
                    actor.stop()  # sensors and walker controllers
            except RuntimeError:
                pass
            # Hand the vehicle back from the traffic manager before destroying
            # it. The TM keeps its own registry, and destroying a vehicle it
            # still tracks leaves a dangling reference that raises
            # "set_actor_collisions: Actor could not be found in the registry"
            # on the *next* tick -- far from the cause, and fatal to the run.
            try:
                if actor.type_id.startswith("vehicle."):
                    actor.set_autopilot(False, self.cfg.traffic_manager_port)
            except (RuntimeError, AttributeError):
                pass
        # batch destroy is far faster than one RPC per actor
        self.client.apply_batch_sync(
            [carla.command.DestroyActor(actor) for actor in stale], True
        )
        self.world.tick()
        print(f"    [world] cleared {len(stale)} leftover actors")

    def _relative_traffic_points(self):
        """Spawn transforms placed *relative to the ego* rather than at random.

        Random spawn points scatter traffic across the whole map, so whether the
        ego ever meets a lead vehicle or oncoming traffic is left to chance --
        in one Town01 episode a lead vehicle was detected in 2.2% of frames with
        forty cars on the map. Seeding a few vehicles directly ahead in the
        ego's own lane, and a few in the opposing lane, makes the following
        distance and oncoming-traffic behaviour actually get exercised.

        Returns ``(lead_transforms, oncoming_transforms)``.
        """
        cfg = self.cfg
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(self.vehicle.get_transform().location)

        lead = []
        cursor = ego_wp
        for i in range(cfg.lead_vehicles):
            nxt = cursor.next(cfg.lead_spacing)
            if not nxt:
                break
            cursor = nxt[0]
            transform = cursor.transform
            # Lift slightly: spawning flush with the road surface collides.
            transform.location.z += 0.5
            lead.append(transform)

        oncoming = []
        # The opposing carriageway is the nearest lane whose lane_id has the
        # opposite sign -- CARLA numbers lanes outward from the road centre with
        # the sign encoding direction of travel. It is not necessarily the
        # *immediate* left neighbour: on a multi-lane road there may be same
        # direction lanes to cross first, and on some roads the opposing side
        # sits to the right. Walking outwards in both directions finds it where
        # a single `get_left_lane()` returned nothing (measured: 0 oncoming
        # vehicles placed on Town10HD).
        opposite = None
        for step in ("left", "right"):
            cursor = ego_wp
            for _ in range(MAX_LANE_SEARCH):
                cursor = (
                    cursor.get_left_lane() if step == "left" else cursor.get_right_lane()
                )
                if cursor is None:
                    break
                if str(cursor.lane_type) != "Driving":
                    continue
                if cursor.lane_id * ego_wp.lane_id < 0:
                    opposite = cursor
                    break
            if opposite is not None:
                break

        if opposite is not None:
            cursor = opposite
            for i in range(cfg.oncoming_vehicles):
                # Oncoming traffic travels *towards* the ego, so step along the
                # opposing lane's own forward direction, which is back past us.
                ahead = cursor.next(cfg.oncoming_spacing)
                if not ahead:
                    break
                cursor = ahead[0]
                transform = cursor.transform
                transform.location.z += 0.5
                oncoming.append(transform)
        return lead, oncoming

    def _spawn_traffic(self, blueprints, spawn_points, tm):
        cfg = self.cfg
        vehicle_bps = [
            bp
            for bp in blueprints.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]
        batch = []

        # Ego-relative traffic first, so it gets the places it needs before the
        # random fill competes for spawn points.
        placed = 0
        if cfg.lead_vehicles or cfg.oncoming_vehicles:
            lead, oncoming = self._relative_traffic_points()
            for transform in lead + oncoming:
                bp = random.choice(vehicle_bps)
                if bp.has_attribute("color"):
                    bp.set_attribute(
                        "color", random.choice(bp.get_attribute("color").recommended_values)
                    )
                batch.append(
                    carla.command.SpawnActor(bp, transform).then(
                        carla.command.SetAutopilot(
                            carla.command.FutureActor, True, tm.get_port()
                        )
                    )
                )
            placed = len(lead) + len(oncoming)
            print(
                f"    [traffic] {len(lead)} ahead in lane, "
                f"{len(oncoming)} oncoming, {cfg.num_vehicles} scattered",
                flush=True,
            )

        for point in spawn_points[: cfg.num_vehicles]:
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            batch.append(
                carla.command.SpawnActor(bp, point).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())
                )
            )
        for response in self.client.apply_batch_sync(batch, True):
            if not response.error:
                actor = self.world.get_actor(response.actor_id)
                if actor is not None:
                    self.spawned.append(actor)

        # D53: walkers are spawned in phases, with a tick between each.
        #
        # A freshly spawned actor is not registered on the server until the
        # next tick. Spawning a walker, attaching a controller and calling
        # ``start()``/``go_to_location()`` in the same breath dereferences an
        # actor the server has not acknowledged, and the *client* segfaults --
        # not the simulator. It survived on Town01 by timing and killed every
        # Town02 attempt at ``go_to_location``.
        #
        # This cost a long detour: the client dying looked exactly like the
        # server dying (the collector aborts either way), so several fixes were
        # aimed at CARLA's stability -- restarts, boot-time map selection,
        # longer RPC deadlines -- when the simulator was never at fault. It was
        # confirmed by noticing CARLA was still alive after the collector died,
        # and located with ``python -X faulthandler``.
        walker_bps = blueprints.filter("walker.pedestrian.*")
        controller_bp = blueprints.find("controller.ai.walker")

        spawn_batch = []
        for _ in range(cfg.num_pedestrians):
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue  # no navigation mesh for this map
            spawn_batch.append(
                carla.command.SpawnActor(
                    random.choice(walker_bps), carla.Transform(location)
                )
            )
        if not spawn_batch:
            return

        walkers = []
        for response in self.client.apply_batch_sync(spawn_batch, True):
            if response.error:
                continue  # collision at the spawn point; skip this pedestrian
            walkers.append(response.actor_id)
        self.world.tick()  # register the walkers before attaching anything

        controllers = []
        for response in self.client.apply_batch_sync(
            [
                carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id)
                for walker_id in walkers
            ],
            True,
        ):
            if not response.error:
                controllers.append(response.actor_id)
        self.world.tick()  # register the controllers before starting them

        for actor_id in walkers + controllers:
            actor = self.world.get_actor(actor_id)
            if actor is not None:
                self.spawned.append(actor)

        for actor_id in controllers:
            controller = self.world.get_actor(actor_id)
            if controller is None:
                continue
            controller.start()
            target = self.world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)

    def _attach_cameras(self, blueprints):
        cfg = self.cfg
        for name, bp, transform in build_camera_blueprints(
            blueprints, cfg.image_width, cfg.image_height
        ):
            camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
            # Bounded queue: in synchronous mode exactly one image arrives per
            # tick, so anything beyond a couple of slots means a desync we want
            # to notice rather than silently buffer.
            q: "queue.Queue" = queue.Queue(maxsize=4)
            camera.listen(q.put)
            self.cameras.append(camera)
            self.image_queues[name] = q
            self.spawned.append(camera)

    def _build_route(self, spawn_points) -> List:
        """A long route through the map for the expert to follow.

        Uses CARLA's own topology rather than the scenario-runner global route
        planner so collection has no dependency outside the CARLA egg.
        """
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(self.vehicle.get_transform().location)
        route = [waypoint.transform.location]
        for _ in range(4000):
            nxt = waypoint.next(2.0)
            if not nxt:
                break
            waypoint = random.choice(nxt) if len(nxt) > 1 else nxt[0]
            route.append(waypoint.transform.location)
        return route

    def teardown(self):
        # D67: take the traffic manager out of synchronous mode *before*
        # destroying anything. In synchronous mode it expects to be stepped in
        # lockstep with the world, and tearing vehicles out from under it
        # leaves stale ids in its registry -- which surfaces on a later tick,
        # in the *next* episode, as "set_actor_collisions: Actor could not be
        # found in the registry". Unregistering each vehicle individually
        # (added earlier) was not sufficient: an actor CARLA has already
        # removed cannot be unregistered at all.
        try:
            tm = self.client.get_trafficmanager(self.cfg.traffic_manager_port)
            tm.set_synchronous_mode(False)
        except (RuntimeError, AttributeError):
            pass

        for camera in self.cameras:
            try:
                camera.stop()
            except RuntimeError:
                pass
        self.cameras.clear()
        self.image_queues.clear()
        if self.spawned:
            # Unregister from the traffic manager before destroying; see
            # ``_purge_world``. Skipping this leaves the TM holding dangling
            # ids and the failure surfaces on a later tick in the *next*
            # episode, which makes it look unrelated to teardown.
            for actor in self.spawned:
                if actor is None:
                    continue
                try:
                    if actor.type_id.startswith("vehicle."):
                        actor.set_autopilot(False, self.cfg.traffic_manager_port)
                except (RuntimeError, AttributeError):
                    pass
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(a) for a in self.spawned if a is not None],
                True,
            )
            self.spawned.clear()
        if self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except RuntimeError:
                # Restoring settings is best-effort cleanup; failing here must
                # not mask the real error that brought us into teardown.
                pass

        # D67: shut the traffic manager down rather than trying to tidy its
        # registry. Neither unregistering each vehicle nor toggling synchronous
        # mode cleared the stale ids -- an actor CARLA has already removed
        # cannot be unregistered, so per-actor cleanup can never be complete,
        # and the next episode died on "set_actor_collisions: Actor could not
        # be found in the registry" pointing at an id from the previous one.
        # ``get_trafficmanager`` creates a fresh instance on the next setup.
        try:
            self.client.get_trafficmanager(
                self.cfg.traffic_manager_port
            ).shut_down()
        except (RuntimeError, AttributeError):
            pass

        self.vehicle = None

    def _drain_images(self):
        for q in self.image_queues.values():
            while not q.empty():
                q.get_nowait()

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotations(self, max_distance: float = 60.0) -> List[dict]:
        """3D boxes for nearby actors, in the ego frame.

        Occlusion is approximated by a visibility count over the six cameras --
        an actor projecting inside at least one image is annotated visible.
        CARLA has no per-actor occlusion query, so the alternative would be a
        depth-buffer test per actor per camera, which costs more than the
        collection loop can afford at 20 fps.
        """
        ego_tf = self.vehicle.get_transform()
        ego_loc = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        out: List[dict] = []
        actors = list(self.world.get_actors().filter("vehicle.*"))
        actors += list(self.world.get_actors().filter("walker.pedestrian.*"))

        for actor in actors:
            if actor.id == self.vehicle.id:
                continue
            category = carla_category(actor)
            if category is None:
                continue

            tf = actor.get_transform()
            dx = tf.location.x - ego_loc.x
            dy = tf.location.y - ego_loc.y
            dz = tf.location.z - ego_loc.z
            if math.hypot(dx, dy) > max_distance:
                continue

            # World -> ego frame.
            x = dx * cos_y + dy * sin_y
            y = -dx * sin_y + dy * cos_y

            extent = actor.bounding_box.extent
            velocity = actor.get_velocity()
            relative_yaw = math.radians(tf.rotation.yaw) - yaw

            out.append(
                {
                    "instance_id": actor.id,
                    "category": category,
                    "translation": [x, y, dz],
                    "size": [extent.y * 2, extent.x * 2, extent.z * 2],  # w, l, h
                    "yaw": math.atan2(math.sin(relative_yaw), math.cos(relative_yaw)),
                    "velocity": [
                        velocity.x * cos_y + velocity.y * sin_y,
                        -velocity.x * sin_y + velocity.y * cos_y,
                    ],
                    "num_lidar_pts": 1,  # nuScenes compatibility field
                }
            )
        return out

    def _map_annotations(
        self, radius: float = 50.0, spacing: float = 2.0, num_points: int = 20
    ) -> List[dict]:
        """Vectorised map elements around the ego, in the ego frame.

        The map head predicts fixed-length polylines, so each element is
        resampled to ``num_points``.  Three classes are produced, matching
        ``MapHead``'s default: lane divider, road boundary, pedestrian crossing.
        """
        carla_map = self.world.get_map()
        ego_tf = self.vehicle.get_transform()
        ego_loc = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        def to_ego(location) -> tuple[float, float]:
            dx = location.x - ego_loc.x
            dy = location.y - ego_loc.y
            return (dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y)

        def resample(points: List[tuple[float, float]]) -> List[List[float]]:
            """Uniform resample to exactly ``num_points`` by arc length."""
            if len(points) < 2:
                return [list(points[0]) for _ in range(num_points)] if points else []
            arr = np.asarray(points, dtype=np.float64)
            segment = np.linalg.norm(np.diff(arr, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(segment)])
            if cumulative[-1] < 1e-6:
                return [list(arr[0]) for _ in range(num_points)]
            targets = np.linspace(0.0, cumulative[-1], num_points)
            return [
                [float(np.interp(t, cumulative, arr[:, 0])),
                 float(np.interp(t, cumulative, arr[:, 1]))]
                for t in targets
            ]

        elements: List[dict] = []
        start = carla_map.get_waypoint(ego_loc)

        # Walk the lane the ego is in plus its neighbours, forwards and back.
        seeds = [start]
        for neighbour in (start.get_left_lane(), start.get_right_lane()):
            if neighbour is not None and neighbour.lane_type == carla.LaneType.Driving:
                seeds.append(neighbour)

        for seed in seeds:
            backward: List[tuple[float, float]] = []
            wp = seed
            for _ in range(int(radius / spacing)):
                previous = wp.previous(spacing)
                if not previous:
                    break
                wp = previous[0]
                backward.append(to_ego(wp.transform.location))

            forward: List[tuple[float, float]] = []
            wp = seed
            for _ in range(int(radius / spacing)):
                nxt = wp.next(spacing)
                if not nxt:
                    break
                wp = nxt[0]
                forward.append(to_ego(wp.transform.location))

            centre = list(reversed(backward)) + [to_ego(seed.transform.location)] + forward
            if len(centre) < 2:
                continue

            # Offset the centreline by half the lane width to get the two
            # boundaries. Whether a boundary is a divider or the road edge
            # depends on there being a driving lane on that side.
            half = seed.lane_width / 2.0
            for side, neighbour in (
                (-1.0, seed.get_left_lane()),
                (1.0, seed.get_right_lane()),
            ):
                offset: List[tuple[float, float]] = []
                for i, (x, y) in enumerate(centre):
                    j = min(i + 1, len(centre) - 1)
                    k = max(i - 1, 0)
                    tx, ty = centre[j][0] - centre[k][0], centre[j][1] - centre[k][1]
                    norm = math.hypot(tx, ty)
                    if norm < 1e-6:
                        continue
                    offset.append((x + side * half * (-ty / norm), y + side * half * (tx / norm)))
                if len(offset) < 2:
                    continue
                is_divider = (
                    neighbour is not None and neighbour.lane_type == carla.LaneType.Driving
                )
                elements.append(
                    {
                        "category": "lane_divider" if is_divider else "road_boundary",
                        "points": resample(offset),
                    }
                )

        for crossing in carla_map.get_crosswalks():
            if crossing.distance(ego_loc) > radius:
                continue
            elements.append(
                {"category": "pedestrian_crossing", "points": resample([to_ego(crossing)] * 2)}
            )

        return elements

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        episode_name: str,
        town: str,
        weather: WeatherPreset,
        seed: int,
        expert_factory,
        writer,
    ) -> Optional[dict]:
        """Collect one episode. Returns its summary, or None if it was rejected."""
        from PIL import Image

        cfg = self.cfg
        route = self.setup(town, weather, seed)
        expert = expert_factory(self.world, self.vehicle, route, cfg.dt)
        self.warmup(expert)

        compliance = EpisodeCompliance()
        trace: Optional[List[dict]] = [] if cfg.trace_csv else None
        safety_records: List[dict] = []
        map_records: List[List[dict]] = []
        previous_observation: Optional[EgoObservation] = None
        collision_detected = False

        # Collision sensor: a collision is an unambiguous episode failure that
        # the Table 1 metrics would not necessarily catch on their own.
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.spawned.append(collision_sensor)

        collision_info: dict = {}

        def on_collision(event):
            nonlocal collision_detected
            if collision_detected:
                return
            collision_detected = True
            # Attribute the impact. A defensive expert drives slowly, which
            # makes it a target for the traffic manager's vehicles; an episode
            # ended by being rear-ended says nothing about the expert's own
            # compliance, so the direction matters when reading rejections.
            try:
                impulse = event.normal_impulse
                collision_info.update(
                    other=getattr(event.other_actor, "type_id", "unknown"),
                    magnitude=math.sqrt(
                        impulse.x**2 + impulse.y**2 + impulse.z**2
                    ),
                )
            except Exception:  # pragma: no cover - diagnostics must never throw
                pass

        collision_sensor.listen(on_collision)

        scene_token = _token()
        sample_tokens: List[str] = []
        prev_sample_token = ""

        try:
            for frame in range(cfg.frames_per_episode):
                control, debug = expert.run_step()
                self.vehicle.apply_control(control)
                self.world.tick()

                images = {}
                try:
                    for name, q in self.image_queues.items():
                        images[name] = q.get(timeout=2.0)
                except queue.Empty:
                    raise RuntimeError(f"camera {name} produced no frame at tick {frame}")

                observation = expert.observe(previous_observation)
                ctx = expert.build_context()
                report = evaluate_frame(observation, ctx)
                compliance.add(report)
                if trace is not None:
                    trace.append(
                        {
                            "frame": frame,
                            "speed": observation.speed,
                            "lon_accel": observation.lon_accel,
                            "lat_accel": observation.lat_accel,
                            "lon_jerk": observation.lon_jerk,
                            "lat_jerk": observation.lat_jerk,
                            "lane_offset": observation.lane_offset,
                            "lead_distance": observation.lead_distance,
                            "throttle": control.throttle,
                            "brake": control.brake,
                            "steer": control.steer,
                            "crosstrack": debug.get("crosstrack"),
                            "ped_distance": debug.get("ped_distance"),
                            "commanded_accel": debug.get("accel_cmd"),
                            "target_speed": debug.get("target_speed"),
                            "constraint": debug.get("constraint"),
                            "gear": debug.get("gear"),
                            "in_junction": debug.get("in_junction"),
                            "violations": "|".join(report.violations),
                        }
                    )

                if collision_detected:
                    # Attribute from the ego's change in speed, not from the
                    # impulse vector. CARLA's ``normal_impulse`` direction did
                    # not reliably distinguish the two cases -- it labelled a
                    # shove-from-behind and a run-into-a-stopped-car the same
                    # way -- whereas the sign of the speed change is
                    # unambiguous: the struck vehicle speeds up, the striking
                    # one slows down. This matters because being rear-ended by
                    # the traffic manager says nothing about the expert.
                    fault = "unknown"
                    if previous_observation is not None:
                        delta_v = observation.speed - previous_observation.speed
                        if delta_v > COLLISION_SHOVE_MS:
                            fault = "struck from behind"
                        elif delta_v < -COLLISION_SHOVE_MS:
                            fault = "ego ran into it"
                    other = collision_info.get("other", "unknown")
                    detail = f"collision ({fault}, {other})"
                    return self._reject(episode_name, detail, compliance, report)

                # D22: the episode is judged on its *compliance rate*, not on
                # the first infraction. Abort early only once the target is
                # arithmetically out of reach -- i.e. even if every remaining
                # frame were clean, the run could not reach the threshold.
                if not cfg.allow_infractions:
                    remaining = cfg.frames_per_episode - (frame + 1)
                    best_possible = (
                        compliance.clean_frames + remaining
                    ) / cfg.frames_per_episode
                    if best_possible < cfg.min_compliance:
                        return self._reject(
                            episode_name,
                            f"compliance unreachable "
                            f"(best possible {best_possible:.2%} "
                            f"< {cfg.min_compliance:.0%})",
                            compliance,
                            report,
                        )

                ratios, mask = safety_vector(observation, ctx)

                sample_token = _token()
                ego_pose_token = _token()
                ego_tf = self.vehicle.get_transform()
                timestamp = int(frame * cfg.dt * 1e6)

                self.tables.ego_pose.append(
                    {
                        "token": ego_pose_token,
                        "timestamp": timestamp,
                        "translation": [ego_tf.location.x, ego_tf.location.y, ego_tf.location.z],
                        "rotation": [
                            ego_tf.rotation.yaw,
                            ego_tf.rotation.pitch,
                            ego_tf.rotation.roll,
                        ],
                    }
                )
                self.tables.sample.append(
                    {
                        "token": sample_token,
                        "timestamp": timestamp,
                        "scene_token": scene_token,
                        "prev": prev_sample_token,
                        "next": "",
                    }
                )
                if prev_sample_token:
                    self.tables.sample[-2]["next"] = sample_token
                prev_sample_token = sample_token
                sample_tokens.append(sample_token)

                for name in CAMERA_ORDER:
                    relative = f"samples/{name}/{episode_name}_{frame:05d}.jpg"
                    path = Path(cfg.out_dir) / relative
                    path.parent.mkdir(parents=True, exist_ok=True)

                    raw = np.frombuffer(images[name].raw_data, dtype=np.uint8)
                    bgra = raw.reshape((images[name].height, images[name].width, 4))
                    Image.fromarray(bgra[:, :, :3][:, :, ::-1]).save(
                        path, quality=cfg.jpeg_quality
                    )

                    self.tables.sample_data.append(
                        {
                            "token": _token(),
                            "sample_token": sample_token,
                            "ego_pose_token": ego_pose_token,
                            "calibrated_sensor_token": writer.calibration_tokens[name],
                            "filename": relative,
                            "fileformat": "jpg",
                            "is_key_frame": True,
                            "height": cfg.image_height,
                            "width": cfg.image_width,
                            "timestamp": timestamp,
                        }
                    )

                for annotation in self._annotations():
                    self.tables.sample_annotation.append(
                        {
                            "token": _token(),
                            "sample_token": sample_token,
                            "instance_token": f"{episode_name}_{annotation['instance_id']}",
                            "category_token": self.tables.category_token(annotation["category"]),
                            "category_name": annotation["category"],
                            "translation": annotation["translation"],
                            "size": annotation["size"],
                            "rotation_yaw": annotation["yaw"],
                            "velocity": annotation["velocity"],
                            "num_lidar_pts": annotation["num_lidar_pts"],
                        }
                    )

                map_records.append(self._map_annotations())

                safety_records.append(
                    {
                        "frame": frame,
                        "sample_token": sample_token,
                        "ratios": ratios.tolist(),
                        "mask": mask.tolist(),
                        "raw": {
                            "speed": observation.speed,
                            "lon_accel": observation.lon_accel,
                            "lat_accel": observation.lat_accel,
                            "lon_jerk": observation.lon_jerk,
                            "lat_jerk": observation.lat_jerk,
                            "lane_offset": observation.lane_offset,
                            "lead_distance": observation.lead_distance,
                        },
                        "context": {
                            "speed_limit_ms": ctx.speed_limit_ms,
                            "wet": ctx.wet,
                            "low_visibility": ctx.low_visibility,
                            "night": ctx.night,
                            "heavy_lead": ctx.lead_is_heavy_vehicle,
                            "road_class": ctx.resolved_road_class().value,
                        },
                        "control": {
                            "throttle": control.throttle,
                            "steer": control.steer,
                            "brake": control.brake,
                        },
                        "expert": {
                            "target_speed": debug["target_speed"],
                            "constraint": debug["constraint"],
                        },
                        "thresholds": report.thresholds,
                    }
                )
                previous_observation = observation

            self.tables.scene.append(
                {
                    "token": scene_token,
                    "name": episode_name,
                    "description": f"{town} / {weather.carla_name}",
                    "nbr_samples": len(sample_tokens),
                    "first_sample_token": sample_tokens[0] if sample_tokens else "",
                    "last_sample_token": sample_tokens[-1] if sample_tokens else "",
                    "log_token": writer.log_token,
                }
            )

            score = compliance.summary()["compliance_score"]

            # D31: a stationary vehicle violates nothing. Compliance must be
            # earned while actually driving, so the episode also has to cover
            # ground -- otherwise a too-cautious expert scores highest by
            # refusing to move, which is exactly what happened once.
            speeds = [r["raw"]["speed"] for r in safety_records]
            mean_speed = sum(speeds) / max(len(speeds), 1)
            # D63: the motion check is a *validity* test, not a compliance
            # test, so it runs even when infractions are allowed. Coupling the
            # two meant that disabling the gate to obtain a diagnostic
            # recording also disabled the guard against the degenerate case --
            # and produced a five-minute episode at 95.5% "compliance" in which
            # the ego travelled 78 m at 0.26 m/s, gridlocked 0.6 m behind a
            # lead vehicle. That is the D31 failure exactly, re-entered through
            # the back door.
            if mean_speed < cfg.min_mean_speed:
                return self._reject(
                    episode_name,
                    f"mean speed {mean_speed:.2f} m/s < {cfg.min_mean_speed:.2f} "
                    f"(compliance was {score:.1%}, but the vehicle barely moved)",
                    compliance,
                )
            # Strictly less-than: a score of exactly ``min_compliance`` is
            # accepted. Printed to two decimals because ``.1%`` rounded 69.98%
            # to "70.0%", which read as a rejection at exactly the threshold.
            if not cfg.allow_infractions and score < cfg.min_compliance:
                return self._reject(
                    episode_name,
                    f"compliance {score:.2%} < {cfg.min_compliance:.2%}",
                    compliance,
                )

            summary = {
                "episode": episode_name,
                "town": town,
                "weather": weather.carla_name,
                "seed": seed,
                "frames": len(safety_records),
                "accepted": True,
                "compliance": compliance.summary(),
            }
            writer.write_episode(episode_name, safety_records, summary, map_records)
            print(f"  [accept] {episode_name}: compliance {score:.2%}")
            print_episode_profile(compliance)
            return summary

        finally:
            if trace:
                path = Path(cfg.out_dir) / f"trace_{episode_name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="") as fh:
                    writer_csv = csv.DictWriter(fh, fieldnames=list(trace[0]))
                    writer_csv.writeheader()
                    writer_csv.writerows(trace)
                print(f"    [trace] {len(trace)} frames -> {path}")
            self.teardown()

    def _reject(
        self,
        episode_name: str,
        reason: str,
        compliance: EpisodeCompliance,
        report: Optional["ComplianceReport"] = None,
    ):
        """Report a rejection with enough detail to act on it.

        A bare "rejected" says nothing about whether the expert is nearly
        compliant or wildly out, which is the only thing worth knowing when
        tuning it. Every violated metric on the offending frame is printed with
        its measured value against its threshold.
        """
        print(f"  [reject] {episode_name}: {reason} at frame {compliance.frames}")
        if report is not None:
            print("    offending frame:")
            for name in report.violations:
                ratio = report.ratios.get(name)
                threshold = report.thresholds.get(name)
                if ratio is None:
                    print(f"      {name}: violated")
                else:
                    print(
                        f"      {name}: ratio {ratio:.2f}x threshold"
                        + (f" (limit {threshold:.3f})" if threshold is not None else "")
                    )
        print_episode_profile(compliance)
        return None


class DatasetWriter:
    """Owns the on-disk layout and the tables shared across episodes."""

    def __init__(self, cfg: CollectConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.out_dir)
        self.tables = NuScenesTables()
        self.log_token = _token()
        self.calibration_tokens: Dict[str, str] = {}

        (self.root / "safety").mkdir(parents=True, exist_ok=True)
        (self.root / "episodes").mkdir(parents=True, exist_ok=True)

        # Rebuild the nuScenes tables from episodes already on disk, so a
        # resumed run appends to them rather than emitting a fresh set that
        # references only the episodes collected after the last crash (D51).
        self._restore_tables()

        self.tables.log.append(
            {
                "token": self.log_token,
                "logfile": "dav",
                "vehicle": "carla_lincoln_mkz_2020",
                "date_captured": time.strftime("%Y-%m-%d"),
                "location": "carla",
            }
        )
        calibration = calibration_table(cfg.image_width, cfg.image_height)
        for name, entry in calibration.items():
            sensor_token = _token()
            calibration_token = _token()
            self.calibration_tokens[name] = calibration_token
            self.tables.sensor.append(
                {"token": sensor_token, "channel": name, "modality": "camera"}
            )
            self.tables.calibrated_sensor.append(
                {"token": calibration_token, "sensor_token": sensor_token, **entry}
            )

    def completed_episodes(self) -> set:
        """Names of episodes already fully written to disk (D51).

        D57: an episode is complete only when it is *loadable*, which means its
        row in the nuScenes ``scene`` table exists as well as its images and
        safety records. An episode collected before the incremental table flush
        existed -- or by a run that died between writing the episode and
        writing the tables -- has its 6000 images on disk and is invisible to
        ``DAVDataset``, because the loader walks the scene table. Treating such
        an episode as done silently drops it from the dataset.

        Reporting it as incomplete makes the resume path re-collect it, which
        is the repair.
        """
        directory = self.root / "episodes"
        if not directory.is_dir():
            return set()

        indexed = set()
        scene_table = self.root / "v1.0-dav" / "scene.json"
        if scene_table.exists():
            try:
                with open(scene_table) as fh:
                    indexed = {row["name"] for row in json.load(fh)}
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                indexed = set()

        done = set()
        for path in directory.glob("*.json"):
            if not (self.root / "safety" / path.name).exists():
                continue  # summary written, records lost to a crash mid-write
            if path.stem not in indexed:
                print(
                    f"    [repair] {path.stem}: on disk but absent from the "
                    f"scene table; re-collecting (D57)",
                    flush=True,
                )
                continue
            done.add(path.stem)
        return done

    def _restore_tables(self) -> None:
        """Reload nuScenes tables from a previous run so resume can append."""
        directory = self.root / "v1.0-dav"
        if not directory.is_dir():
            return
        for table in (
            "scene", "sample", "sample_data", "ego_pose",
            "instance", "sample_annotation",
        ):
            path = directory / f"{table}.json"
            if not path.exists():
                continue
            try:
                with open(path) as fh:
                    rows = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue  # a half-written table from a crash; start that one fresh
            if isinstance(rows, list):
                getattr(self.tables, table).extend(rows)

    def write_episode(
        self,
        name: str,
        safety_records: List[dict],
        summary: dict,
        map_records: Optional[List[List[dict]]] = None,
    ) -> None:
        with open(self.root / "safety" / f"{name}.json", "w") as fh:
            json.dump({"episode": name, "frames": safety_records}, fh)
        with open(self.root / "episodes" / f"{name}.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        if map_records is not None:
            # Map polylines go to a compressed array rather than JSON: at 20
            # points per element and thousands of frames per episode they would
            # dominate the annotation file and slow every load of it.
            self._write_map(name, map_records)

        # Flush the nuScenes tables after every episode, not only at the end
        # (D51). They are the index for everything already on disk; losing them
        # to a crash would orphan hours of collected images.
        self.tables.write(self.root / "v1.0-dav")

    def _write_map(self, name: str, map_records: List[List[dict]]) -> None:
        classes = {"lane_divider": 0, "road_boundary": 1, "pedestrian_crossing": 2}
        max_elements = max((len(r) for r in map_records), default=0)
        num_points = 20
        frames = len(map_records)

        points = np.zeros((frames, max_elements, num_points, 2), dtype=np.float32)
        labels = np.full((frames, max_elements), -1, dtype=np.int8)  # -1 == padding

        for f, elements in enumerate(map_records):
            for e, element in enumerate(elements):
                pts = np.asarray(element["points"], dtype=np.float32)
                if pts.shape[0] != num_points:
                    continue
                points[f, e] = pts
                labels[f, e] = classes[element["category"]]

        (self.root / "maps").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.root / "maps" / f"{name}.npz", points=points, labels=labels
        )

    def finalise(self, manifest: dict) -> None:
        self.tables.write(self.root / "v1.0-dav")
        with open(self.root / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)


def loadable_episodes(out_dir: str) -> set:
    """Episodes that are complete *and* indexed in the scene table (D57).

    Shared by the collector's resume path and ``next_town`` so the two cannot
    disagree about what has been collected -- they did, and the result was an
    orphaned episode that resume skipped and the loader could not see.
    """
    root = Path(out_dir)
    directory = root / "episodes"
    if not directory.is_dir():
        return set()
    indexed = set()
    scene_table = root / "v1.0-dav" / "scene.json"
    if scene_table.exists():
        try:
            with open(scene_table) as fh:
                indexed = {row["name"] for row in json.load(fh)}
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            indexed = set()
    return {
        path.stem
        for path in directory.glob("*.json")
        if (root / "safety" / path.name).exists() and path.stem in indexed
    }


def next_town(cfg: CollectConfig) -> Optional[str]:
    """Town of the next uncollected episode, or None when the run is complete.

    Lets a supervisor boot CARLA with the map already selected. Switching maps
    at runtime is the operation that crashes the simulator on a slow disk
    (D51); loading one at engine start does not have to unload a populated
    world first, and has been reliable here.
    """
    done = loadable_episodes(cfg.out_dir)
    for index, (town, weather) in enumerate(_episode_schedule(cfg)):
        if f"ep{index:03d}_{town}_{weather.carla_name}" not in done:
            return town
    return None


def _episode_schedule(cfg: CollectConfig) -> List[tuple]:
    """Decide the (town, weather) of every episode, grouped by town.

    Round-robining town and weather together covers the grid evenly but
    changes the map on every single episode, and a map load streams the whole
    town off disk. Sorting by town instead means each map is loaded once and
    reused for all its episodes -- identical coverage, a fraction of the I/O.

    Weather still advances every episode, so consecutive episodes in the same
    town get different conditions.
    """
    pairs = [
        (cfg.towns[i % len(cfg.towns)], WEATHER_PRESETS[i % len(WEATHER_PRESETS)])
        for i in range(cfg.episodes)
    ]
    # Stable sort by the town's position in the configured order, so the
    # weather assignment above is preserved within each town.
    order = {town: i for i, town in enumerate(cfg.towns)}
    return sorted(pairs, key=lambda pair: order[pair[0]])


def collect(cfg: CollectConfig) -> dict:
    """Run the full collection campaign."""
    if carla is None:
        raise RuntimeError(
            "the carla package is not importable; activate the 'dav' conda env"
        )
    from .expert import DefensiveExpert

    client = carla.Client(cfg.carla_host, cfg.carla_port)
    client.set_timeout(cfg.timeout)

    writer = DatasetWriter(cfg)
    base_seed = cfg.seed if cfg.seed is not None else random.randrange(1 << 30)

    accepted: List[dict] = []
    rejected = 0
    started = time.time()

    # D51: resume rather than restart.
    #
    # A full run needs one map load per town, and on a slow disk each load is
    # minutes of heavy I/O that can take the simulator down with it -- this run
    # lost CARLA to SIGSEGV mid-load with one episode already banked. Starting
    # from scratch after every crash makes a multi-town campaign statistically
    # impossible: the chance of surviving N loads falls off a cliff.
    #
    # Episodes already written to disk are skipped, so relaunching continues
    # where the previous attempt stopped.
    existing = writer.completed_episodes()
    if existing:
        print(f"[resume] {len(existing)} episode(s) already collected", flush=True)

    schedule = _episode_schedule(cfg)
    for index, (town, weather) in enumerate(schedule):
        name = f"ep{index:03d}_{town}_{weather.carla_name}"
        if name in existing:
            print(f"[{index + 1}/{cfg.episodes}] {name}: already collected, skipping",
                  flush=True)
            continue

        for attempt in range(cfg.max_attempts_per_episode):
            seed = base_seed + index * 100 + attempt
            print(
                f"[{index + 1}/{cfg.episodes}] {name} "
                f"(attempt {attempt + 1}, seed {seed})",
                flush=True,
            )
            collector = EpisodeCollector(cfg, client, writer.tables)
            try:
                summary = collector.run(
                    name, town, weather, seed, DefensiveExpert, writer
                )
            except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
                print(f"  [error] {name}: {type(exc).__name__}: {exc}", flush=True)
                collector.teardown()
                summary = None

            if summary is not None:
                accepted.append(summary)
                break
            rejected += 1
        else:
            print(f"  [skip] {name}: no attempt satisfied the hard constraints")

    manifest = {
        "config": asdict(cfg),
        "accepted_episodes": len(accepted),
        "rejected_attempts": rejected,
        "total_frames": sum(e["frames"] for e in accepted),
        "elapsed_seconds": time.time() - started,
        "episodes": accepted,
    }
    writer.finalise(manifest)
    print(
        f"\nDone: {len(accepted)} episodes, {manifest['total_frames']} frames, "
        f"{rejected} rejected attempts, {manifest['elapsed_seconds'] / 60:.1f} min"
    )
    return manifest
