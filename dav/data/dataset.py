"""Torch dataset over the collected defensive-driving episodes.

Reads the nuScenes-shaped tables written by ``collector.py`` and produces the
tensors ``SafetyAwareDriveTransformer.forward`` expects.

Two details worth stating up front.

**Safety inputs are lagged by one frame.**  The safety query is conditioned on
the *previous* frame's measured ratios, never the current one.  Conditioning on
the current frame would hand the model the answer to what it is being asked to
predict, and the lag is what makes the same input available in closed loop.
See DOCUMENTATION.md, deviation D5.

**Safety history is re-encoded from scalars, not cached embeddings.**  Thesis
modification 5 attends over "the corresponding safety query embedding" for the
past N timesteps.  Caching per-frame embeddings from a network that is still
training makes them stale within an epoch.  Instead the past N frames' scalar
ratio vectors are loaded and passed through the *current* safety encoder, so
the history is always consistent with the weights that produced it and carries
gradient.  See DOCUMENTATION.md, deviation D11.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..metrics.thresholds import NUM_SAFETY_METRICS
from .sensors import CAMERA_ORDER

#: ImageNet statistics -- the timm backbones are pretrained with these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAP_CLASSES = ("lane_divider", "road_boundary", "pedestrian_crossing")


class DAVDataset(Dataset):
    """One item == one keyframe, with its history and its future.

    Parameters
    ----------
    root
        Dataset directory written by ``collect``.
    horizon
        Number of future waypoints to supervise the planner with.
    waypoint_stride
        Frames between consecutive waypoints. Collection runs at 20 Hz, so a
        stride of 1 would make ``horizon`` 6 waypoints cover 0.3 s -- far too
        short to express a plan, and short enough that the planning loss
        degenerates into predicting "keep going". A stride of 10 gives 0.5 s
        spacing and a 3 s horizon, the interval nuScenes and DriveTransformer
        both use. Scale it with the collection frame rate.
    history
        N, the FIFO depth for safety temporal cross-attention.
    image_size
        ``(H, W)`` the images are resized to. Must match the model config.
    max_agents / max_map
        Padding widths for the annotation tensors.
    episodes
        Restrict to these episode names (used for the train/val split).
    """

    def __init__(
        self,
        root: str | Path,
        horizon: int = 6,
        history: int = 4,
        image_size: tuple[int, int] = (224, 400),
        max_agents: int = 64,
        max_map: int = 32,
        episodes: Optional[Sequence[str]] = None,
        augment: bool = False,
        waypoint_stride: int = 10,
    ) -> None:
        self.root = Path(root)
        self.horizon = horizon
        self.waypoint_stride = max(int(waypoint_stride), 1)
        self.history = history
        self.image_size = tuple(image_size)
        self.max_agents = max_agents
        self.max_map = max_map
        self.augment = augment

        tables = self.root / "v1.0-dav"
        if not tables.exists():
            raise FileNotFoundError(
                f"no dataset tables at {tables}; run scripts/collect_data.sh first"
            )

        self.samples = _load(tables / "sample.json")
        self.sample_data = _load(tables / "sample_data.json")
        self.ego_poses = {p["token"]: p for p in _load(tables / "ego_pose.json")}
        self.scenes = _load(tables / "scene.json")
        self.categories = _load(tables / "category.json")
        self.category_index = {c["token"]: i for i, c in enumerate(self.categories)}

        annotations = _load(tables / "sample_annotation.json")
        self.annotations_by_sample: Dict[str, List[dict]] = {}
        for a in annotations:
            self.annotations_by_sample.setdefault(a["sample_token"], []).append(a)

        # Images, keyed by (sample_token, camera).
        self.images_by_sample: Dict[str, Dict[str, dict]] = {}
        self.pose_by_sample: Dict[str, dict] = {}
        for sd in self.sample_data:
            channel = Path(sd["filename"]).parent.name
            self.images_by_sample.setdefault(sd["sample_token"], {})[channel] = sd
            self.pose_by_sample[sd["sample_token"]] = self.ego_poses[sd["ego_pose_token"]]

        # Episode -> ordered sample tokens, plus the per-frame safety records.
        self.episode_names: List[str] = []
        self.safety: Dict[str, List[dict]] = {}
        self.maps: Dict[str, Dict[str, np.ndarray]] = {}
        self.index: List[tuple[str, int]] = []

        wanted = set(episodes) if episodes is not None else None
        for scene in self.scenes:
            name = scene["name"]
            if wanted is not None and name not in wanted:
                continue
            safety_path = self.root / "safety" / f"{name}.json"
            if not safety_path.exists():
                continue
            records = _load(safety_path)["frames"]
            self.safety[name] = records
            self.episode_names.append(name)

            map_path = self.root / "maps" / f"{name}.npz"
            if map_path.exists():
                # D58: decompress now rather than holding the archive open.
                #
                # ``np.load`` on an .npz returns a lazy handle onto an open zip
                # file. ``DataLoader`` forks its workers, every worker inherits
                # the same file descriptors, and their interleaved reads corrupt
                # each other's decompressor state -- which surfaces as
                # ``zlib.error: invalid block type`` from a file that is
                # perfectly intact on disk. It only appears with num_workers > 0
                # *and* real map data, so the synthetic dataset used for earlier
                # testing never reached it.
                #
                # The arrays are small (~0.5 MB per episode here) and loading
                # them before the fork means the workers share one copy-on-write
                # copy, so this costs less memory than it appears to.
                with np.load(map_path) as archive:
                    self.maps[name] = {
                        "points": archive["points"],
                        "labels": archive["labels"],
                    }

            # Frames near the end have no complete future trajectory, so they
            # cannot supervise the planner and are excluded.
            usable = len(records) - horizon * self.waypoint_stride
            self.index += [(name, i) for i in range(max(usable, 0))]

        if not self.index:
            raise RuntimeError(f"no usable frames found under {self.root}")

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------

    def _load_images(self, sample_token: str) -> torch.Tensor:
        h, w = self.image_size
        views = self.images_by_sample.get(sample_token, {})
        out = np.zeros((len(CAMERA_ORDER), 3, h, w), dtype=np.float32)

        for i, camera in enumerate(CAMERA_ORDER):
            entry = views.get(camera)
            if entry is None:
                continue  # leaves a zero image; the camera embedding still marks the view
            image = Image.open(self.root / entry["filename"]).convert("RGB")
            if image.size != (w, h):
                image = image.resize((w, h), Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
            out[i] = array.transpose(2, 0, 1)

        return torch.from_numpy(out)

    def _future_trajectory(self, records: List[dict], frame: int) -> torch.Tensor:
        """Future ego waypoints in the current ego frame, ``[horizon, 2]``.

        Poses are stored in world coordinates, so each future position is
        rotated into the frame of the current pose.
        """
        current = self.pose_by_sample[records[frame]["sample_token"]]
        cx, cy, _ = current["translation"]
        yaw = math.radians(current["rotation"][0])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        points = np.zeros((self.horizon, 2), dtype=np.float32)
        for k in range(self.horizon):
            index = frame + (k + 1) * self.waypoint_stride
            future = self.pose_by_sample[records[index]["sample_token"]]
            dx = future["translation"][0] - cx
            dy = future["translation"][1] - cy
            points[k] = (dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y)
        return torch.from_numpy(points)

    def _agents(self, sample_token: str) -> Dict[str, torch.Tensor]:
        annotations = self.annotations_by_sample.get(sample_token, [])[: self.max_agents]
        boxes = np.zeros((self.max_agents, 8), dtype=np.float32)
        labels = np.full((self.max_agents,), -1, dtype=np.int64)
        velocity = np.zeros((self.max_agents, 2), dtype=np.float32)

        for i, a in enumerate(annotations):
            x, y, z = a["translation"]
            w, l, h = a["size"]
            yaw = a["rotation_yaw"]
            boxes[i] = [x, y, z, w, l, h, math.sin(yaw), math.cos(yaw)]
            labels[i] = self.category_index.get(a["category_token"], 0)
            velocity[i] = a["velocity"]

        return {
            "agent_boxes_gt": torch.from_numpy(boxes),
            "agent_labels_gt": torch.from_numpy(labels),
            "agent_velocity_gt": torch.from_numpy(velocity),
        }

    def _map(self, episode: str, frame: int) -> Dict[str, torch.Tensor]:
        points = np.zeros((self.max_map, 20, 2), dtype=np.float32)
        labels = np.full((self.max_map,), -1, dtype=np.int64)

        npz = self.maps.get(episode)
        if npz is not None and frame < npz["points"].shape[0]:
            src_points = npz["points"][frame]
            src_labels = npz["labels"][frame]
            n = min(src_points.shape[0], self.max_map)
            points[:n] = src_points[:n]
            labels[:n] = src_labels[:n].astype(np.int64)

        return {
            "map_points_gt": torch.from_numpy(points),
            "map_labels_gt": torch.from_numpy(labels),
        }

    def _safety_history(
        self, records: List[dict], frame: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[N, 6]`` ratios, ``[N, 6]`` metric masks, ``[N]`` padding mask.

        Padding mask follows the torch attention convention: True == ignore.
        Frames before the start of the episode are padding.
        """
        ratios = np.zeros((self.history, NUM_SAFETY_METRICS), dtype=np.float32)
        masks = np.zeros((self.history, NUM_SAFETY_METRICS), dtype=np.float32)
        padding = np.ones((self.history,), dtype=bool)

        for k in range(self.history):
            # Oldest first: slot 0 is frame - history, slot N-1 is frame - 1.
            source = frame - self.history + k
            if source < 0:
                continue
            ratios[k] = records[source]["ratios"]
            masks[k] = records[source]["mask"]
            padding[k] = False

        return (
            torch.from_numpy(ratios),
            torch.from_numpy(masks),
            torch.from_numpy(padding),
        )

    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode, frame = self.index[index]
        records = self.safety[episode]
        record = records[frame]
        sample_token = record["sample_token"]

        raw = record["raw"]
        ego_state = torch.tensor(
            [
                raw["speed"],
                raw["lon_accel"],
                raw["lat_accel"],
                record["control"]["steer"],
                raw["lon_jerk"],
                raw["lat_jerk"],
            ],
            dtype=torch.float32,
        )

        # Safety input is the *previous* frame's measurement (see D5). On the
        # first frame of an episode there is none, and valid=0 makes the
        # encoder substitute its null token.
        if frame > 0:
            previous = records[frame - 1]
            safety_in = torch.tensor(previous["ratios"], dtype=torch.float32)
            safety_in_mask = torch.tensor(previous["mask"], dtype=torch.float32)
            safety_valid = torch.tensor(1.0)
        else:
            safety_in = torch.zeros(NUM_SAFETY_METRICS)
            safety_in_mask = torch.zeros(NUM_SAFETY_METRICS)
            safety_valid = torch.tensor(0.0)

        history_ratios, history_masks, history_padding = self._safety_history(
            records, frame
        )

        item: Dict[str, torch.Tensor] = {
            "images": self._load_images(sample_token),
            "ego_state": ego_state,
            "command": torch.tensor(_command_from(record), dtype=torch.long),
            "safety_ratios": safety_in,
            "safety_mask": safety_in_mask,
            "safety_valid": safety_valid,
            "safety_history_ratios": history_ratios,
            "safety_history_masks": history_masks,
            "safety_history_padding": history_padding,
            # Target: the *current* frame's measured ratios, which is what the
            # safety head is asked to predict.
            "safety_target": torch.tensor(record["ratios"], dtype=torch.float32),
            "safety_target_mask": torch.tensor(record["mask"], dtype=torch.float32),
            "trajectory_gt": self._future_trajectory(records, frame),
        }
        item.update(self._agents(sample_token))
        item.update(self._map(episode, frame))
        return item

    # ------------------------------------------------------------------

    @staticmethod
    def split(
        root: str | Path, val_fraction: float = 0.15, seed: int = 0
    ) -> tuple[List[str], List[str]]:
        """Episode-level train/val split.

        Split by episode, never by frame: consecutive frames of one episode are
        near-duplicates, so a frame-level split leaks the validation set into
        training and reports a meaninglessly low validation loss.
        """
        scenes = _load(Path(root) / "v1.0-dav" / "scene.json")
        names = sorted(s["name"] for s in scenes)
        rng = np.random.default_rng(seed)
        rng.shuffle(names)
        n_val = max(1, int(len(names) * val_fraction))
        return names[n_val:], names[:n_val]


def _load(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _command_from(record: dict) -> int:
    """High-level navigation command index.

    The collector's expert follows a free route rather than a scenario-runner
    plan, so there is no recorded command. It is inferred from the binding
    constraint and steering, which is enough for the command embedding to carry
    useful signal. Indices follow CARLA's RoadOption ordering:
    0 VOID, 1 LEFT, 2 RIGHT, 3 STRAIGHT, 4 LANEFOLLOW, 5 CHANGELANELEFT,
    6 CHANGELANERIGHT.
    """
    steer = record["control"]["steer"]
    if record["expert"]["constraint"] in ("red_light", "yellow_light", "pedestrian"):
        return 3
    if steer < -0.15:
        return 1
    if steer > 0.15:
        return 2
    return 4
