"""Training entry point.

Writes a JSONL metric stream that the GUI tails live, plus checkpoints and a
run manifest.  Run with::

    python -m dav.train --config configs/dav_tiny.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import DAVDataset
from .losses.criterion import DAVCriterion
from .losses.safety_huber import SafetyViolationMetrics
from .models.drive_transformer import DAVConfig, build_model
from .utils.logging import RunLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_dataloaders(cfg: dict, model_cfg: DAVConfig):
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    train_names, val_names = DAVDataset.split(
        root, data_cfg.get("val_fraction", 0.15), data_cfg.get("split_seed", 0)
    )

    common = dict(
        root=root,
        horizon=model_cfg.horizon,
        history=model_cfg.temporal_length,
        image_size=model_cfg.image_size,
        max_agents=data_cfg.get("max_agents", 64),
        max_map=data_cfg.get("max_map", 32),
        waypoint_stride=data_cfg.get("waypoint_stride", 10),
    )
    train_set = DAVDataset(episodes=train_names, augment=True, **common)
    val_set = DAVDataset(episodes=val_names, augment=False, **common)

    loader_args = dict(
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=data_cfg.get("num_workers", 4) > 0,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, **loader_args
    )
    loader_args["drop_last"] = False
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
        shuffle=False, **loader_args
    )
    return train_loader, val_loader, train_names, val_names


def cosine_lr(step: int, total: int, base: float, warmup: int, floor: float = 0.02) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    scale = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base * scale


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype) -> Dict[str, float]:
    model.eval()
    violation = SafetyViolationMetrics()
    totals: Dict[str, float] = {}
    batches = 0

    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
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
            _, logs = criterion(outputs, batch)

        if "pred_safety_ratios" in outputs:
            logs.update(
                violation(
                    outputs["pred_safety_ratios"].float(),
                    batch["safety_target"],
                    batch["safety_target_mask"],
                )
            )
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        batches += 1

    model.train()
    return {f"val/{k}": v / max(batches, 1) for k, v in totals.items()}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the safety-aware DriveTransformer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="dotted config overrides, e.g. train.batch_size=2",
    )
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    for override in args.overrides:
        key, _, value = override.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)

    model_cfg = DAVConfig(**cfg["model"])
    train_cfg = cfg["train"]

    set_seed(train_cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or cfg.get("name") or Path(args.config).stem
    run_name = f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    train_loader, val_loader, train_names, val_names = build_dataloaders(cfg, model_cfg)

    model = build_model(model_cfg).to(device)
    criterion = DAVCriterion(
        num_agent_classes=model_cfg.num_agent_classes,
        num_map_classes=model_cfg.num_map_classes,
        weights=cfg.get("loss_weights"),
        safety_delta=cfg.get("safety", {}).get("delta", 0.1),
        safety_alpha=cfg.get("safety", {}).get("alpha", 5.0),
        literal_thesis_huber=cfg.get("safety", {}).get("literal_thesis_formula", False),
    ).to(device)

    # Backbone gets a lower learning rate: it is pretrained and the rest of the
    # network is not, so a shared rate either destroys the backbone or starves
    # the transformer.
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.body") else other_params).append(param)

    base_lr = float(train_cfg["lr"])
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * train_cfg.get("backbone_lr_mult", 0.1)},
            {"params": other_params, "lr": base_lr},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    precision = train_cfg.get("precision", "amp")
    amp_dtype = {"amp": torch.float16, "bf16": torch.bfloat16, "fp32": None}[precision]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    epochs = int(train_cfg["epochs"])
    accum = int(train_cfg.get("grad_accum", 1))
    steps_per_epoch = max(len(train_loader) // accum, 1)
    total_steps = steps_per_epoch * epochs
    warmup = int(train_cfg.get("warmup_steps", min(500, total_steps // 10)))

    logger = RunLogger(run_dir)
    logger.write_manifest(
        {
            "run": run_name,
            "config": cfg,
            "model_parameters": model.num_parameters(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "train_episodes": train_names,
            "val_episodes": val_names,
            "train_frames": len(train_loader.dataset),
            "val_frames": len(val_loader.dataset),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "started": time.time(),
            "status": "running",
        }
    )

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_val = state.get("best_val", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    stop = {"requested": False}

    def handle_signal(_signum, _frame):
        # Checkpoint on Ctrl-C rather than losing the epoch.
        stop["requested"] = True
        print("\ninterrupt received; finishing the current step then saving", flush=True)

    signal.signal(signal.SIGINT, handle_signal)

    print(
        f"run {run_name} | {model.num_parameters() / 1e6:.1f}M params | "
        f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val frames | "
        f"{total_steps} steps",
        flush=True,
    )

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.time()
        optimiser.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            batch = move(batch, device)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
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
                loss, logs = criterion(outputs, batch)

            scaler.scale(loss / accum).backward()

            if (i + 1) % accum == 0:
                lr = cosine_lr(global_step, total_steps, base_lr, warmup)
                for group, mult in zip(
                    optimiser.param_groups, (train_cfg.get("backbone_lr_mult", 0.1), 1.0)
                ):
                    group["lr"] = lr * mult

                scaler.unscale_(optimiser)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.get("grad_clip", 1.0)
                )
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % train_cfg.get("log_every", 10) == 0:
                    logs.update(
                        {
                            "lr": lr,
                            "grad_norm": float(grad_norm),
                            "epoch": epoch,
                            "step": global_step,
                            "gpu_mb": (
                                torch.cuda.max_memory_allocated() / 1e6
                                if device.type == "cuda"
                                else 0.0
                            ),
                        }
                    )
                    logger.log(logs)
                    print(
                        f"e{epoch} s{global_step}/{total_steps} "
                        f"loss {logs['loss/total']:.4f} "
                        f"plan {logs['loss/plan']:.4f} "
                        f"safety {logs.get('loss/safety', 0):.4f} "
                        f"ade {logs['metric/ade']:.3f} lr {lr:.2e}",
                        flush=True,
                    )

            if stop["requested"]:
                break

        val_logs = validate(model, val_loader, criterion, device, amp_dtype)
        val_logs.update({"epoch": epoch, "step": global_step,
                         "epoch_seconds": time.time() - epoch_started})
        logger.log(val_logs)
        print(
            f"[val] epoch {epoch}: total {val_logs['val/loss/total']:.4f} "
            f"ade {val_logs['val/metric/ade']:.3f}",
            flush=True,
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "model_config": asdict(model_cfg),
        }
        torch.save(checkpoint, run_dir / "last.pt")

        score = val_logs["val/loss/total"]
        if score < best_val:
            best_val = score
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, run_dir / "best.pt")
            logger.log({"event": "best_checkpoint", "epoch": epoch, "val_total": score})
            print(f"  new best ({score:.4f}) -> {run_dir / 'best.pt'}", flush=True)

        if stop["requested"]:
            break

    logger.update_manifest({"status": "interrupted" if stop["requested"] else "finished",
                            "finished": time.time(), "best_val": best_val})
    print(f"\nrun directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
