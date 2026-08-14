"""Closed-loop driving agent, conforming to the CARLA leaderboard interface.

Bench2Drive runs on the CARLA Leaderboard 2.0 harness, which imports a module
exposing ``get_entry_point()`` and an ``AutonomousAgent`` subclass.  The base
class lives in ``leaderboard.autoagents.autonomous_agent``, which is only
importable when the Bench2Drive repository is on ``PYTHONPATH``; when it is
absent this module falls back to a local stub so the agent can still be
imported, unit-tested and driven by ``dav.eval_b2d`` without the harness.

The interesting part is ``_safety_input``: at training time the safety query is
conditioned on the previous frame's measured metric ratios, and this is where
that same quantity gets produced from live sensors instead of an annotation
file.  Five of the six come from the IMU and speedometer; the following gap
comes from the model's own previous-step estimate.  See DOCUMENTATION.md, D5.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.sensors import CAMERA_ORDER, CAMERA_RIG
from ..metrics.evaluator import EgoObservation, evaluate_frame, safety_vector
from ..metrics.thresholds import (
    NUM_SAFETY_METRICS,
    SAFETY_VECTOR_ORDER,
    BRAKE_MAX_EMERGENCY,
    DrivingContext,
    following_gap_for,
)
from ..models.drive_transformer import DAVConfig, build_model

try:  # pragma: no cover - only present inside the Bench2Drive harness
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
except ImportError:  # pragma: no cover

    class Track:  # minimal stand-in
        SENSORS = "SENSORS"

    class AutonomousAgent:  # minimal stand-in
        def __init__(self, *_args, **_kwargs):
            self.track = Track.SENSORS

        def setup(self, path_to_conf_file):
            raise NotImplementedError

        def sensors(self):
            raise NotImplementedError

        def run_step(self, input_data, timestamp):
            raise NotImplementedError

        def destroy(self):
            pass


def get_entry_point() -> str:
    """Required by the leaderboard loader."""
    return "DAVAgent"


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DAVAgent(AutonomousAgent):
    """Safety-aware DriveTransformer driving in closed loop."""

    def setup(self, path_to_conf_file: str) -> None:
        import yaml

        self.track = Track.SENSORS

        with open(path_to_conf_file) as fh:
            conf = yaml.safe_load(fh)

        self.device = torch.device(conf.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(conf["checkpoint"], map_location=self.device, weights_only=False)
        self.cfg = DAVConfig(**checkpoint["model_config"])
        self.model = build_model(self.cfg).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"])

        self.waypoint_dt = float(conf.get("waypoint_dt", 0.5))
        self.control_dt = float(conf.get("control_dt", 0.05))
        self.save_frames_to: Optional[Path] = (
            Path(conf["save_frames_to"]) if conf.get("save_frames_to") else None
        )
        if self.save_frames_to:
            self.save_frames_to.mkdir(parents=True, exist_ok=True)

        # Rolling state for the safety input and for jerk, which needs the
        # previous acceleration.
        self.history: deque = deque(maxlen=self.cfg.temporal_length)
        self.previous_observation: Optional[EgoObservation] = None
        self.previous_speed = 0.0
        self.previous_accel = (0.0, 0.0)
        self.estimated_gap_ratio = 0.0
        self.step = 0
        self.frame_log: List[dict] = []

        self.controller = LongitudinalLateralController(self.control_dt)

    # ------------------------------------------------------------------

    def sensors(self) -> List[dict]:
        """Sensor rig, in the leaderboard's declaration format."""
        height, width = self.cfg.image_size
        out: List[dict] = []
        for spec in CAMERA_RIG:
            out.append(
                {
                    "type": "sensor.camera.rgb",
                    "id": spec.name,
                    "x": spec.x, "y": spec.y, "z": spec.z,
                    "roll": spec.roll, "pitch": spec.pitch, "yaw": spec.yaw,
                    "width": width, "height": height, "fov": spec.fov,
                }
            )
        out += [
            {"type": "sensor.other.imu", "id": "IMU",
             "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.05},
            {"type": "sensor.speedometer", "id": "SPEED", "reading_frequency": 20},
            {"type": "sensor.other.gnss", "id": "GPS",
             "x": 0.0, "y": 0.0, "z": 0.0, "sensor_tick": 0.01},
        ]
        return out

    # ------------------------------------------------------------------

    def _images(self, input_data: dict) -> torch.Tensor:
        height, width = self.cfg.image_size
        batch = np.zeros((len(CAMERA_ORDER), 3, height, width), dtype=np.float32)

        for i, name in enumerate(CAMERA_ORDER):
            entry = input_data.get(name)
            if entry is None:
                continue
            # The leaderboard hands over (frame_number, BGRA array).
            array = entry[1]
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3][:, :, ::-1]
            if array.shape[:2] != (height, width):
                from PIL import Image

                array = np.asarray(
                    Image.fromarray(array.astype(np.uint8)).resize(
                        (width, height), Image.BILINEAR
                    )
                )
            array = array.astype(np.float32) / 255.0
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
            batch[i] = array.transpose(2, 0, 1)

        return torch.from_numpy(batch).unsqueeze(0).to(self.device)

    def _safety_input(self, input_data: dict, speed: float) -> tuple:
        """Previous-frame violation ratios, from live sensors.

        This is the inference-time counterpart of the annotation file the
        thesis conditions on. Acceleration comes from the IMU, jerk from
        differencing it, speed from the speedometer -- all causally available.
        The following gap alone cannot be measured without perception, so the
        model's own previous-step estimate stands in, flagged as such.
        """
        imu = input_data.get("IMU")
        if imu is not None:
            reading = imu[1]
            # IMU accelerometer is [ax, ay, az] in the ego frame.
            lon_accel = float(reading[0])
            lat_accel = float(reading[1])
        else:
            lon_accel = (speed - self.previous_speed) / self.control_dt
            lat_accel = 0.0

        lon_jerk = (lon_accel - self.previous_accel[0]) / self.control_dt
        lat_jerk = (lat_accel - self.previous_accel[1]) / self.control_dt

        observation = EgoObservation(
            speed=speed,
            lon_accel=lon_accel,
            lat_accel=lat_accel,
            lon_jerk=lon_jerk,
            lat_jerk=lat_jerk,
            lane_offset=0.0,  # needs a map; not part of the six loss metrics
        )
        # Speed limit is not exposed through the leaderboard sensor set, so the
        # nominal urban limit is assumed. It scales only the speed ratio.
        ctx = DrivingContext(speed_limit_ms=13.9)
        ratios, mask = safety_vector(observation, ctx)

        gap_index = SAFETY_VECTOR_ORDER.index("following_gap")
        if self.estimated_gap_ratio > 0.0:
            ratios[gap_index] = self.estimated_gap_ratio
            mask[gap_index] = 1.0

        self.previous_accel = (lon_accel, lat_accel)
        self.previous_speed = speed
        self.previous_observation = observation
        return ratios, mask, observation

    def _history_tensors(self):
        n = self.cfg.temporal_length
        ratios = np.zeros((n, NUM_SAFETY_METRICS), dtype=np.float32)
        masks = np.zeros((n, NUM_SAFETY_METRICS), dtype=np.float32)
        padding = np.ones((n,), dtype=bool)

        items = list(self.history)
        # Oldest first, right-aligned so the newest lands in the last slot --
        # the same layout the dataset produces.
        for k, (r, m) in enumerate(items):
            slot = n - len(items) + k
            ratios[slot] = r
            masks[slot] = m
            padding[slot] = False

        to = lambda a: torch.from_numpy(a).unsqueeze(0).to(self.device)  # noqa: E731
        return to(ratios), to(masks), to(padding)

    # ------------------------------------------------------------------

    def run_step(self, input_data: dict, timestamp: float):
        import carla

        speed_entry = input_data.get("SPEED")
        speed = float(speed_entry[1]["speed"]) if speed_entry else 0.0

        ratios, mask, observation = self._safety_input(input_data, speed)
        history_ratios, history_masks, history_padding = self._history_tensors()

        images = self._images(input_data)
        ego_state = torch.tensor(
            [[speed, observation.lon_accel, observation.lat_accel,
              self.controller.previous_steer, observation.lon_jerk, observation.lat_jerk]],
            dtype=torch.float32, device=self.device,
        )
        command = torch.tensor([4], dtype=torch.long, device=self.device)  # LANEFOLLOW

        with torch.no_grad():
            outputs = self.model(
                images=images,
                ego_state=ego_state,
                command=command,
                safety_ratios=torch.from_numpy(ratios).unsqueeze(0).to(self.device),
                safety_mask=torch.from_numpy(mask).unsqueeze(0).to(self.device),
                safety_valid=torch.tensor(
                    [1.0 if self.step > 0 else 0.0], device=self.device
                ),
                safety_history_ratios=history_ratios,
                safety_history_masks=history_masks,
                safety_history_mask=history_padding,
            )

        trajectory = outputs["trajectory"][0].float().cpu().numpy()
        predicted_ratios = (
            outputs["pred_safety_ratios"][0].float().cpu().numpy()
            if "pred_safety_ratios" in outputs
            else np.zeros(NUM_SAFETY_METRICS, dtype=np.float32)
        )
        # Feed the model's own following-gap estimate back in for the next step.
        self.estimated_gap_ratio = float(
            predicted_ratios[SAFETY_VECTOR_ORDER.index("following_gap")]
        )

        control = self.controller.step(trajectory, speed, self.waypoint_dt, predicted_ratios)

        self.history.append((ratios.copy(), mask.copy()))
        self.frame_log.append(
            {
                "step": self.step,
                "timestamp": timestamp,
                "speed": speed,
                "measured_ratios": ratios.tolist(),
                "predicted_ratios": predicted_ratios.tolist(),
                "worst_measured": float((ratios * mask).max()) if mask.any() else 0.0,
                "control": {"throttle": control.throttle, "steer": control.steer,
                            "brake": control.brake},
            }
        )
        self.step += 1
        return control

    def destroy(self) -> None:
        self.history.clear()
        self.frame_log.clear()


class LongitudinalLateralController:
    """Turns predicted waypoints into throttle/steer/brake.

    Kept deliberately simple and, importantly, *defensive*: the target speed is
    capped by the model's own predicted safety ratios, so a trajectory the
    safety head thinks is unsafe is executed more slowly rather than as-is.
    That is the inference-time expression of the whole thesis -- without it,
    the safety query would only ever influence driving through the trajectory
    shape.
    """

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.previous_steer = 0.0
        self.integral = 0.0

    def step(self, trajectory: np.ndarray, speed: float, waypoint_dt: float,
             safety_ratios: np.ndarray):
        import carla

        if len(trajectory) == 0:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        # Target speed from the first waypoint's displacement over its interval.
        first = trajectory[0]
        target_speed = float(np.linalg.norm(first) / max(waypoint_dt, 1e-3))

        # Safety governor: scale the target down by how far the worst predicted
        # ratio exceeds its threshold.
        worst = float(np.max(safety_ratios)) if safety_ratios.size else 0.0
        if worst > 1.0:
            target_speed /= min(worst, 3.0)

        error = target_speed - speed
        self.integral = float(np.clip(self.integral + error * self.dt, -5.0, 5.0))
        accel = 1.2 * error + 0.15 * self.integral
        accel = float(np.clip(accel, -BRAKE_MAX_EMERGENCY, 3.0))

        if accel >= 0:
            throttle, brake = min(accel / 3.0, 0.75), 0.0
        else:
            throttle, brake = 0.0, min(abs(accel) / BRAKE_MAX_EMERGENCY, 1.0)

        # Pure-pursuit steering onto a lookahead point on the predicted path.
        lookahead = trajectory[min(1, len(trajectory) - 1)]
        steer = float(np.clip(math.atan2(lookahead[1], max(lookahead[0], 1e-3)) / 0.6, -1, 1))
        # Rate limit, which bounds lateral jerk exactly as the expert's does.
        steer = float(np.clip(steer, self.previous_steer - 0.15, self.previous_steer + 0.15))
        self.previous_steer = steer

        return carla.VehicleControl(
            throttle=float(throttle), steer=steer, brake=float(brake)
        )
