"""The six-camera rig and its calibration.

Chapter 3: "the model requires multi-view images (6 cameras: front, front left,
front right, rear, rear left and rear right), 3D position of the vehicle,
camera parameters and ego vehicle state".

Camera names and placement follow nuScenes so the dataset drops into tooling
built for it, which is the stated reason for mirroring that format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    """One camera: nuScenes-style name, mount pose, and intrinsics.

    Mount pose is in the ego frame using CARLA's convention: x forward, y
    right, z up, yaw positive clockwise seen from above, all in metres/degrees.
    """

    name: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0
    fov: float = 70.0


#: Ring of six cameras. The front camera gets a narrower FOV for range, the
#: side and rear cameras a wider one for coverage -- the same trade nuScenes
#: makes with its 70 degree ring and 110 degree rear camera.
CAMERA_RIG: tuple[CameraSpec, ...] = (
    CameraSpec("CAM_FRONT", 1.5, 0.0, 1.6, 0.0, fov=70.0),
    CameraSpec("CAM_FRONT_LEFT", 1.3, -0.5, 1.6, -55.0, fov=70.0),
    CameraSpec("CAM_FRONT_RIGHT", 1.3, 0.5, 1.6, 55.0, fov=70.0),
    CameraSpec("CAM_BACK", -1.6, 0.0, 1.6, 180.0, fov=110.0),
    CameraSpec("CAM_BACK_LEFT", -0.8, -0.5, 1.6, -110.0, fov=70.0),
    CameraSpec("CAM_BACK_RIGHT", -0.8, 0.5, 1.6, 110.0, fov=70.0),
)

#: Order the model expects along the camera axis of its input tensor.
CAMERA_ORDER: tuple[str, ...] = tuple(c.name for c in CAMERA_RIG)


def intrinsic_matrix(width: int, height: int, fov_degrees: float) -> np.ndarray:
    """Pinhole intrinsics for a CARLA RGB camera.

    CARLA's ``fov`` is the *horizontal* field of view, and its cameras are
    square-pixel with the principal point at the image centre, so a single
    focal length follows from the width alone.
    """
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """CARLA (yaw, pitch, roll) in degrees -> 3x3 rotation, ego frame."""
    y, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def quaternion_from_euler(yaw: float, pitch: float, roll: float) -> List[float]:
    """(yaw, pitch, roll) degrees -> ``[w, x, y, z]``, the nuScenes convention."""
    y, p, r = (math.radians(v) / 2.0 for v in (yaw, pitch, roll))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def calibration_table(width: int, height: int) -> Dict[str, dict]:
    """Per-camera calibration in the nuScenes ``calibrated_sensor`` shape."""
    table: Dict[str, dict] = {}
    for spec in CAMERA_RIG:
        table[spec.name] = {
            "translation": [spec.x, spec.y, spec.z],
            "rotation": quaternion_from_euler(spec.yaw, spec.pitch, spec.roll),
            "camera_intrinsic": intrinsic_matrix(width, height, spec.fov).tolist(),
            "fov": spec.fov,
            "width": width,
            "height": height,
        }
    return table


def build_camera_blueprints(blueprint_library, width: int, height: int):
    """CARLA blueprints + transforms for the rig.

    Returns a list of ``(name, blueprint, carla.Transform)``.  Imported lazily
    so this module can be used (for calibration maths and tests) without CARLA.
    """
    import carla

    out = []
    for spec in CAMERA_RIG:
        bp = blueprint_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(spec.fov))
        # Match the simulation step so every camera fires exactly once per tick.
        bp.set_attribute("sensor_tick", "0.0")
        transform = carla.Transform(
            carla.Location(x=spec.x, y=spec.y, z=spec.z),
            carla.Rotation(yaw=spec.yaw, pitch=spec.pitch, roll=spec.roll),
        )
        out.append((spec.name, bp, transform))
    return out
