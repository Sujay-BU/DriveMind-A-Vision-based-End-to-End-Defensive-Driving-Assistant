#!/usr/bin/env python
"""Render the best episode to a single MP4 in a separate output directory.

"Best" is whichever episode scores highest under ``--rank-by``:

    compliance   fraction of frames with no Table 1 infraction  (default)
    ade          lowest mean trajectory error against the expert
    driving      highest Bench2Drive driving score, where an eval was run

Two frame sources:

    --source dataset   replay a collected episode through a checkpoint. Needs
                       no CARLA process, so it works anywhere the dataset and
                       a checkpoint exist. This is the default.
    --source rollout   use frames dumped by the Bench2Drive harness during a
                       closed-loop run (``--frames-dir``).

Output is one MP4 plus a JSON sidecar the dashboard reads:

    python scripts/export_best_video.py \
        --run runs/dav_tiny_20260810-2129 --data data/dav_pilot \
        --out-dir outputs/videos
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dav.data.dataset import DAVDataset
from dav.data.sensors import CAMERA_ORDER
from dav.metrics.thresholds import SAFETY_VECTOR_ORDER
from dav.models.drive_transformer import DAVConfig, build_model

# Mosaic layout: forward-facing row on top, rearward row below, each in the
# left-to-right order a driver would scan.
MOSAIC = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]

METRIC_LABELS = {
    "lon_accel": "lon accel",
    "lat_accel": "lat accel",
    "lon_jerk": "lon jerk",
    "lat_jerk": "lat jerk",
    "following_gap": "follow gap",
    "speed": "speed",
}

# Palette matched to the dashboard so the two read as one product.
INK = (11, 11, 11)
SURFACE = (252, 252, 251)
PANEL = (244, 244, 241)
MUTED = (137, 135, 129)
GRID = (225, 224, 217)
BLUE = (42, 120, 214)
ORANGE = (235, 104, 52)
GOOD = (12, 163, 12)
CRITICAL = (208, 59, 59)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load_font_bold(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Episode ranking
# ---------------------------------------------------------------------------


def rank_episodes(
    data_root: Path, rank_by: str, eval_path: Optional[Path]
) -> List[Tuple[str, float, dict]]:
    """Every episode with its score, best first."""
    episodes_dir = data_root / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"no episodes directory under {data_root}")

    driving_scores: Dict[str, float] = {}
    if eval_path and eval_path.exists():
        with open(eval_path) as fh:
            payload = json.load(fh)
        for route in payload.get("routes", []):
            driving_scores[route.get("name", "")] = route.get("driving_score", 0.0)

    scored: List[Tuple[str, float, dict]] = []
    for path in sorted(episodes_dir.glob("*.json")):
        with open(path) as fh:
            summary = json.load(fh)
        name = summary.get("episode", path.stem)
        compliance = summary.get("compliance", {})

        if rank_by == "compliance":
            score = compliance.get("compliance_score", 0.0)
        elif rank_by == "driving":
            score = driving_scores.get(name, 0.0)
        else:  # ade -- filled in later, after inference
            score = 0.0
        scored.append((name, float(score), summary))

    return sorted(scored, key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """``[3, H, W]`` normalised tensor -> ``[H, W, 3]`` uint8."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    array = tensor.detach().cpu().numpy() * std + mean
    return (np.clip(array, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def build_mosaic(images: torch.Tensor, tile: Tuple[int, int]) -> Image.Image:
    """``[N_cam, 3, H, W]`` -> a 3x2 contact sheet with camera captions."""
    tw, th = tile
    sheet = Image.new("RGB", (tw * 3, th * 2), SURFACE)
    draw = ImageDraw.Draw(sheet)
    font = load_font(13)

    for row, names in enumerate(MOSAIC):
        for col, name in enumerate(names):
            index = CAMERA_ORDER.index(name)
            view = Image.fromarray(denormalise(images[index])).resize(
                (tw, th), Image.BILINEAR
            )
            sheet.paste(view, (col * tw, row * th))
            x, y = col * tw + 8, row * th + 6
            label = name.replace("CAM_", "").replace("_", " ").lower()
            # Shadow first so the caption survives a bright sky.
            draw.text((x + 1, y + 1), label, font=font, fill=(0, 0, 0))
            draw.text((x, y), label, font=font, fill=(255, 255, 255))

    for col in range(1, 3):
        draw.line([(col * tw, 0), (col * tw, th * 2)], fill=SURFACE, width=2)
    draw.line([(0, th), (tw * 3, th)], fill=SURFACE, width=2)
    return sheet


def draw_trajectory_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    predicted: np.ndarray,
    expert: np.ndarray,
) -> None:
    """Bird's-eye plot of the predicted trajectory against the expert's.

    Ego frame: +x forward, +y right. Drawn with forward pointing up, which is
    how a driver reads it, so the plot's vertical axis is x and its horizontal
    axis is -y.
    """
    x0, y0, x1, y1 = box
    font = load_font(11)
    font_bold = load_font_bold(12)

    draw.rectangle(box, fill=PANEL, outline=GRID)
    draw.text((x0 + 10, y0 + 8), "PLANNED PATH", font=font_bold, fill=MUTED)

    pad = 28
    plot = (x0 + pad, y0 + pad, x1 - pad, y1 - 14)
    px0, py0, px1, py1 = plot
    width, height = px1 - px0, py1 - py0

    points = np.concatenate([predicted, expert], axis=0) if len(expert) else predicted
    if not len(points):
        return
    forward_max = max(float(points[:, 0].max()), 5.0)
    lateral_max = max(float(np.abs(points[:, 1]).max()), 2.5)

    def to_pixels(pt: np.ndarray) -> Tuple[float, float]:
        fx = pt[0] / forward_max
        ly = pt[1] / lateral_max
        return (px0 + width / 2 - ly * (width / 2), py1 - fx * height)

    # Reference grid: the ego's own position and the straight-ahead axis.
    draw.line([(px0 + width / 2, py0), (px0 + width / 2, py1)], fill=GRID, width=1)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        y = py1 - fraction * height
        draw.line([(px0, y), (px1, y)], fill=GRID, width=1)
        draw.text((px1 + 3, y - 6), f"{forward_max * fraction:.0f}m", font=font, fill=MUTED)

    if len(expert) >= 2:
        draw.line([to_pixels(p) for p in expert], fill=ORANGE, width=3, joint="curve")
    if len(predicted) >= 2:
        draw.line([to_pixels(p) for p in predicted], fill=BLUE, width=3, joint="curve")
    for p in predicted:
        cx, cy = to_pixels(p)
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=BLUE)

    # Ego marker.
    ex, ey = to_pixels(np.array([0.0, 0.0]))
    draw.polygon([(ex, ey - 7), (ex - 5, ey + 5), (ex + 5, ey + 5)], fill=INK)

    legend_y = y1 - 12
    draw.rectangle([x0 + 10, legend_y - 4, x0 + 20, legend_y + 4], fill=BLUE)
    draw.text((x0 + 25, legend_y - 7), "model", font=font, fill=INK)
    draw.rectangle([x0 + 75, legend_y - 4, x0 + 85, legend_y + 4], fill=ORANGE)
    draw.text((x0 + 90, legend_y - 7), "expert", font=font, fill=INK)


def draw_safety_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    ratios: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Horizontal bars of the six violation ratios, with the threshold marked.

    1.0 is the Table 1 threshold; a bar past the dashed line is an infraction,
    and is coloured critical rather than relying on length alone.
    """
    x0, y0, x1, y1 = box
    font = load_font(11)
    font_bold = load_font_bold(12)

    draw.rectangle(box, fill=PANEL, outline=GRID)
    draw.text((x0 + 10, y0 + 8), "DEFENSIVE-DRIVING METRICS", font=font_bold, fill=MUTED)

    label_width = 76
    bar_x0 = x0 + 12 + label_width
    bar_x1 = x1 - 52
    # The scale runs to 2x threshold so a compliant bar sits at half width and
    # the dashed threshold line lands mid-panel where it is easy to read.
    scale = 2.0
    threshold_x = bar_x0 + (bar_x1 - bar_x0) / scale

    top = y0 + 30
    row_height = (y1 - top - 12) / len(SAFETY_VECTOR_ORDER)
    bar_height = min(int(row_height * 0.55), 14)

    for i, name in enumerate(SAFETY_VECTOR_ORDER):
        cy = top + row_height * (i + 0.5)
        draw.text((x0 + 12, cy - 6), METRIC_LABELS[name], font=font, fill=INK)

        if mask[i] == 0.0:
            draw.text((bar_x0, cy - 6), "n/a", font=font, fill=MUTED)
            continue

        ratio = float(ratios[i])
        filled = min(ratio / scale, 1.0)
        colour = CRITICAL if ratio > 1.0 else GOOD
        draw.rectangle(
            [bar_x0, cy - bar_height / 2, bar_x1, cy + bar_height / 2],
            fill=(232, 231, 226),
        )
        if filled > 0:
            draw.rectangle(
                [bar_x0, cy - bar_height / 2, bar_x0 + filled * (bar_x1 - bar_x0),
                 cy + bar_height / 2],
                fill=colour,
            )
        draw.text((bar_x1 + 6, cy - 6), f"{ratio:.2f}", font=font, fill=INK)

    # Dashed threshold line across the whole stack.
    y = top
    while y < y1 - 12:
        draw.line([(threshold_x, y), (threshold_x, min(y + 4, y1 - 12))], fill=INK, width=1)
        y += 8
    draw.text((threshold_x - 16, y1 - 14), "1.0", font=font, fill=INK)


def draw_status_panel(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    info: Dict[str, object],
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=PANEL, outline=GRID)
    font_small = load_font(11)
    font_value = load_font_bold(21)
    font_label = load_font(11)

    entries = [
        ("SPEED", f"{info['speed_kmh']:.0f}", "km/h"),
        ("ADE", f"{info['ade']:.2f}", "m"),
        ("WORST RATIO", f"{info['worst_ratio']:.2f}", "x threshold"),
        ("FRAME", f"{info['frame']}", f"of {info['total_frames']}"),
    ]
    cell = (x1 - x0) / len(entries)
    for i, (label, value, unit) in enumerate(entries):
        cx = x0 + cell * i + 14
        draw.text((cx, y0 + 10), label, font=font_small, fill=MUTED)
        colour = CRITICAL if (label == "WORST RATIO" and float(value) > 1.0) else INK
        draw.text((cx, y0 + 26), value, font=font_value, fill=colour)
        draw.text((cx + 4 + font_value.getlength(value), y0 + 36), unit,
                  font=font_label, fill=MUTED)

    status = "COMPLIANT" if info["compliant"] else "INFRACTION"
    colour = GOOD if info["compliant"] else CRITICAL
    text_width = font_value.getlength(status)
    draw.rectangle([x1 - text_width - 34, y0 + 18, x1 - 14, y0 + 48], outline=colour, width=2)
    draw.text((x1 - text_width - 24, y0 + 22), status, font=load_font_bold(15), fill=colour)


def compose_frame(
    images: torch.Tensor,
    predicted: np.ndarray,
    expert: np.ndarray,
    ratios: np.ndarray,
    mask: np.ndarray,
    info: Dict[str, object],
    title: str,
    tile: Tuple[int, int],
) -> np.ndarray:
    tw, th = tile
    mosaic = build_mosaic(images, tile)
    header = 40
    panel_height = 230
    width = tw * 3
    height = header + th * 2 + panel_height

    canvas = Image.new("RGB", (width, height), SURFACE)
    canvas.paste(mosaic, (0, header))
    draw = ImageDraw.Draw(canvas)

    draw.text((14, 12), title, font=load_font_bold(15), fill=INK)
    draw.text((width - 14 - load_font(12).getlength("DAV · safety-aware DriveTransformer"), 15),
              "DAV · safety-aware DriveTransformer", font=load_font(12), fill=MUTED)

    top = header + th * 2
    status_height = 62
    draw_status_panel(draw, (10, top + 6, width - 10, top + 6 + status_height), info)

    body_top = top + 6 + status_height + 8
    body_bottom = height - 10
    split = int(width * 0.42)
    draw_trajectory_panel(draw, (10, body_top, split - 5, body_bottom), predicted, expert)
    draw_safety_panel(draw, (split + 5, body_top, width - 10, body_bottom), ratios, mask)

    # ffmpeg's H.264 encoder requires even dimensions.
    if width % 2 or height % 2:
        canvas = canvas.crop((0, 0, width - width % 2, height - height % 2))
    return np.asarray(canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the best episode as one MP4")
    parser.add_argument("--run", required=True, help="run directory containing best.pt")
    parser.add_argument("--data", required=True, help="dataset root")
    parser.add_argument("--out-dir", default="outputs/videos",
                        help="separate directory the video is written to")
    parser.add_argument("--source", choices=("dataset", "rollout"), default="dataset")
    parser.add_argument("--frames-dir", default=None, help="for --source rollout")
    parser.add_argument("--rank-by", choices=("compliance", "ade", "driving"),
                        default="compliance")
    parser.add_argument("--episode", default=None, help="override the ranking")
    parser.add_argument("--max-frames", type=int, default=900,
                        help="frames to render; 20 fps, so 6000 == 5 minutes")
    parser.add_argument("--name", default=None,
                        help="output filename stem, overriding <run>__<episode>")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing video instead of refusing")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--checkpoint", default=None, help="defaults to <run>/best.pt")
    parser.add_argument("--waypoint-stride", type=int, default=None,
                        help="defaults to the value recorded in the run's config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run)
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint_path}")

    device = torch.device(args.device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = DAVConfig(**state["model_config"])
    model = build_model(model_cfg).to(device).eval()
    model.load_state_dict(state["model"])

    ranked = rank_episodes(data_root, args.rank_by, run_dir / "bench2drive.json")
    if not ranked:
        raise RuntimeError(f"no episodes found under {data_root}")
    if args.episode:
        ranked = [r for r in ranked if r[0] == args.episode] or ranked
    episode, score, summary = ranked[0]
    print(f"episode {episode}  ({args.rank_by} score {score:.4f})")

    # Waypoint spacing has to match what the checkpoint was trained with, or
    # the plotted expert path is on a different time scale than the model's.
    stride = args.waypoint_stride
    if stride is None:
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as fh:
                stride = (yaml.safe_load(fh).get("data") or {}).get("waypoint_stride", 10)
        else:
            stride = 10

    dataset = DAVDataset(
        root=data_root,
        horizon=model_cfg.horizon,
        history=model_cfg.temporal_length,
        image_size=model_cfg.image_size,
        episodes=[episode],
        waypoint_stride=stride,
    )
    total = min(len(dataset), args.max_frames)
    if total == 0:
        raise RuntimeError(f"episode {episode} has no usable frames")

    tile_width = args.tile_width
    tile_height = int(tile_width * model_cfg.image_size[0] / model_cfg.image_size[1])
    tile = (tile_width, tile_height)

    import imageio.v2 as imageio

    stem = args.name or f"{run_dir.name}__{episode}"
    output = out_dir / f"{stem}.mp4"
    if output.exists() and not args.force:
        # Re-exporting the same run and episode -- a longer take, a different
        # checkpoint -- silently replaced the earlier file, which is a poor way
        # to lose a recording that took an hour to produce. Refuse by default
        # and say what to do about it.
        raise SystemExit(
            f"{output} already exists.\n"
            f"Pass --name <stem> to write alongside it, or --force to replace it."
        )
    writer = imageio.get_writer(
        output, fps=args.fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_log_level="error",
    )

    ade_sum = 0.0
    violation_frames = 0

    with torch.no_grad():
        for i in range(total):
            item = dataset[i]
            batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}

            outputs = model(
                images=batch["images"],
                ego_state=batch["ego_state"],
                command=batch["command"],
                safety_ratios=batch["safety_ratios"],
                safety_mask=batch["safety_mask"],
                safety_valid=batch["safety_valid"],
                safety_history_ratios=batch["safety_history_ratios"],
                safety_history_masks=batch["safety_history_masks"],
                safety_history_mask=batch["safety_history_padding"],
            )

            predicted = outputs["trajectory"][0].float().cpu().numpy()
            expert = item["trajectory_gt"].numpy()
            ade = float(np.linalg.norm(predicted - expert, axis=-1).mean())
            ade_sum += ade

            # Show the *measured* ratios: what the vehicle actually did on this
            # frame, which is the honest safety readout. The model's prediction
            # is a separate quantity and is charted in the dashboard.
            ratios = item["safety_target"].numpy()
            mask = item["safety_target_mask"].numpy()
            worst = float((ratios * mask).max()) if mask.any() else 0.0
            compliant = worst <= 1.0
            if not compliant:
                violation_frames += 1

            frame = compose_frame(
                images=item["images"],
                predicted=predicted,
                expert=expert,
                ratios=ratios,
                mask=mask,
                info={
                    "speed_kmh": float(item["ego_state"][0]) * 3.6,
                    "ade": ade,
                    "worst_ratio": worst,
                    "compliant": compliant,
                    "frame": i,
                    "total_frames": total,
                },
                title=f"{episode}  ·  {run_dir.name}",
                tile=tile,
            )
            writer.append_data(frame)

            if i % 50 == 0:
                print(f"  frame {i}/{total}", flush=True)

    writer.close()

    meta = {
        "episode": episode,
        "run": run_dir.name,
        "checkpoint": str(checkpoint_path),
        "ranked_by": args.rank_by,
        "rank_score": score,
        "frames": total,
        "fps": args.fps,
        "mean_ade": ade_sum / total,
        "compliance_score": 1.0 - violation_frames / total,
        "town": summary.get("town"),
        "weather": summary.get("weather"),
    }
    with open(output.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    size_mb = output.stat().st_size / 1e6
    print(
        f"\nwrote {output}  ({size_mb:.1f} MB, {total} frames)\n"
        f"  mean ADE {meta['mean_ade']:.3f} m · "
        f"compliance {meta['compliance_score'] * 100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
