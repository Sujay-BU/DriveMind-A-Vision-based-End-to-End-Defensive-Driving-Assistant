#!/usr/bin/env python
"""Generate a small dataset in the exact on-disk format the collector writes.

Purpose is verification, not training: it exercises every reader in the
pipeline (tables, images, safety records, map arrays) so that ``dav.train``,
the GUI and the video exporter can be tested without a CARLA run.  A model
trained on this will learn nothing useful -- the images are procedural noise.

    python scripts/make_synthetic_dataset.py --out data/dav_synth \
        --episodes 4 --frames 60
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dav.data.collector import CATEGORIES
from dav.data.sensors import CAMERA_ORDER, calibration_table
from dav.metrics.evaluator import EgoObservation, evaluate_frame, safety_vector
from dav.metrics.thresholds import TOWNS, WEATHER_PRESETS, DrivingContext


def token() -> str:
    return uuid.uuid4().hex


def synthetic_frame(width: int, height: int, phase: float, camera: int) -> Image.Image:
    """A cheap deterministic gradient so frames differ and compress like images."""
    xs = np.linspace(0, 1, width, dtype=np.float32)
    ys = np.linspace(0, 1, height, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    r = (np.sin(6 * gx + phase + camera) * 0.5 + 0.5) * 255
    g = (np.cos(5 * gy + phase * 0.7) * 0.5 + 0.5) * 255
    b = np.full_like(r, (camera * 40) % 255)
    return Image.fromarray(np.stack([r, g, b], axis=-1).astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dav_synth")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=225)
    args = parser.parse_args()

    root = Path(args.out)
    tables_dir = root / "v1.0-dav"
    for sub in ("safety", "episodes", "maps"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    category = [{"token": token(), "name": n, "description": n} for n in CATEGORIES]
    category_tokens = {c["name"]: c["token"] for c in category}

    sensor, calibrated_sensor = [], []
    calibration_tokens = {}
    for name, entry in calibration_table(args.width, args.height).items():
        st, ct = token(), token()
        calibration_tokens[name] = ct
        sensor.append({"token": st, "channel": name, "modality": "camera"})
        calibrated_sensor.append({"token": ct, "sensor_token": st, **entry})

    log_token = token()
    log = [{"token": log_token, "logfile": "dav-synth", "vehicle": "synthetic",
            "date_captured": time.strftime("%Y-%m-%d"), "location": "synthetic"}]

    scene, sample, sample_data, ego_pose, sample_annotation, instance = [], [], [], [], [], []

    for e in range(args.episodes):
        town = TOWNS[e % len(TOWNS)]
        weather = WEATHER_PRESETS[e % len(WEATHER_PRESETS)]
        name = f"ep{e:03d}_{town}_{weather.carla_name}"
        scene_token = token()
        tokens, records = [], []
        map_points = np.zeros((args.frames, 6, 20, 2), dtype=np.float32)
        map_labels = np.full((args.frames, 6), -1, dtype=np.int8)

        x = y = heading = 0.0
        speed = 8.0
        previous = None

        for f in range(args.frames):
            # A plausible ego trajectory: gentle S-curve at roughly constant speed.
            heading += 0.01 * math.sin(f * 0.05)
            speed += float(rng.normal(0, 0.05))
            speed = float(np.clip(speed, 3.0, 14.0))
            x += speed * 0.05 * math.cos(heading)
            y += speed * 0.05 * math.sin(heading)

            lon_accel = float(rng.normal(0, 0.4))
            lat_accel = float(rng.normal(0, 0.5))
            observation = EgoObservation(
                speed=speed,
                lon_accel=lon_accel,
                lat_accel=lat_accel,
                lon_jerk=0.0 if previous is None else (lon_accel - previous.lon_accel) / 0.05,
                lat_jerk=0.0 if previous is None else (lat_accel - previous.lat_accel) / 0.05,
                lane_offset=abs(float(rng.normal(0, 0.08))),
                lead_distance=float(rng.uniform(20, 60)) if f % 3 else None,
            )
            ctx = DrivingContext(
                speed_limit_ms=13.9,
                wet=weather.wet,
                low_visibility=weather.low_visibility,
                night=weather.night,
            )
            ratios, mask = safety_vector(observation, ctx)
            report = evaluate_frame(observation, ctx)

            sample_token, pose_token = token(), token()
            timestamp = int(f * 0.05 * 1e6)

            ego_pose.append({"token": pose_token, "timestamp": timestamp,
                             "translation": [x, y, 0.0],
                             "rotation": [math.degrees(heading), 0.0, 0.0]})
            sample.append({"token": sample_token, "timestamp": timestamp,
                           "scene_token": scene_token,
                           "prev": tokens[-1] if tokens else "", "next": ""})
            if tokens:
                sample[-2]["next"] = sample_token
            tokens.append(sample_token)

            for c, camera in enumerate(CAMERA_ORDER):
                relative = f"samples/{camera}/{name}_{f:05d}.jpg"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                synthetic_frame(args.width, args.height, f * 0.1, c).save(path, quality=85)
                sample_data.append({
                    "token": token(), "sample_token": sample_token,
                    "ego_pose_token": pose_token,
                    "calibrated_sensor_token": calibration_tokens[camera],
                    "filename": relative, "fileformat": "jpg", "is_key_frame": True,
                    "height": args.height, "width": args.width, "timestamp": timestamp,
                })

            for a in range(int(rng.integers(2, 8))):
                sample_annotation.append({
                    "token": token(), "sample_token": sample_token,
                    "instance_token": f"{name}_{a}",
                    "category_token": category_tokens["vehicle.car"],
                    "category_name": "vehicle.car",
                    "translation": [float(rng.uniform(-40, 40)),
                                    float(rng.uniform(-15, 15)), 0.0],
                    "size": [1.8, 4.5, 1.6],
                    "rotation_yaw": float(rng.uniform(-math.pi, math.pi)),
                    "velocity": [float(rng.uniform(-5, 12)), 0.0],
                    "num_lidar_pts": 1,
                })

            for m in range(6):
                offset = (m - 2.5) * 3.5
                along = np.linspace(-10, 40, 20)
                map_points[f, m, :, 0] = along
                map_points[f, m, :, 1] = offset + 0.02 * along
                map_labels[f, m] = m % 3

            records.append({
                "frame": f, "sample_token": sample_token,
                "ratios": ratios.tolist(), "mask": mask.tolist(),
                "raw": {"speed": observation.speed, "lon_accel": observation.lon_accel,
                        "lat_accel": observation.lat_accel, "lon_jerk": observation.lon_jerk,
                        "lat_jerk": observation.lat_jerk,
                        "lane_offset": observation.lane_offset,
                        "lead_distance": observation.lead_distance},
                "context": {"speed_limit_ms": ctx.speed_limit_ms, "wet": ctx.wet,
                            "low_visibility": ctx.low_visibility, "night": ctx.night,
                            "heavy_lead": False,
                            "road_class": ctx.resolved_road_class().value},
                "control": {"throttle": 0.4, "steer": float(np.clip(heading, -1, 1)),
                            "brake": 0.0},
                "expert": {"target_speed": speed, "constraint": "speed_limit"},
                "thresholds": report.thresholds,
            })
            previous = observation

        scene.append({"token": scene_token, "name": name,
                      "description": f"{town} / {weather.carla_name} (synthetic)",
                      "nbr_samples": len(tokens),
                      "first_sample_token": tokens[0], "last_sample_token": tokens[-1],
                      "log_token": log_token})

        with open(root / "safety" / f"{name}.json", "w") as fh:
            json.dump({"episode": name, "frames": records}, fh)
        with open(root / "episodes" / f"{name}.json", "w") as fh:
            json.dump({"episode": name, "town": town, "weather": weather.carla_name,
                       "seed": 0, "frames": len(records), "accepted": True,
                       "synthetic": True}, fh, indent=2)
        np.savez_compressed(root / "maps" / f"{name}.npz",
                            points=map_points, labels=map_labels)
        print(f"  {name}: {len(records)} frames")

    for table_name, rows in [
        ("scene", scene), ("sample", sample), ("sample_data", sample_data),
        ("ego_pose", ego_pose), ("calibrated_sensor", calibrated_sensor),
        ("sensor", sensor), ("instance", instance),
        ("sample_annotation", sample_annotation), ("log", log), ("category", category),
    ]:
        with open(tables_dir / f"{table_name}.json", "w") as fh:
            json.dump(rows, fh)

    with open(root / "manifest.json", "w") as fh:
        json.dump({"synthetic": True, "accepted_episodes": args.episodes,
                   "total_frames": args.episodes * args.frames}, fh, indent=2)

    print(f"\nsynthetic dataset at {root}: {args.episodes} episodes, "
          f"{args.episodes * args.frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
