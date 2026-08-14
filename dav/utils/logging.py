"""Run logging: a JSONL metric stream plus a manifest, both tailed by the GUI.

JSONL rather than TensorBoard event files so the dashboard can read the stream
with the standard library and so a run remains inspectable with ``tail -f``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class RunLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self._handle = open(self.metrics_path, "a", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        record.setdefault("time", time.time())
        self._handle.write(json.dumps(_jsonable(record)) + "\n")

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        _atomic_write(self.manifest_path, manifest)

    def update_manifest(self, patch: Dict[str, Any]) -> None:
        current = read_manifest(self.run_dir) or {}
        current.update(patch)
        _atomic_write(self.manifest_path, current)

    def close(self) -> None:
        self._handle.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    """Write via a temp file + rename so a reader never sees a half-written file.

    The GUI polls the manifest while training writes it; a plain open/write
    would occasionally hand the dashboard truncated JSON.
    """
    directory = path.parent
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_manifest(run_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None  # mid-write; the caller polls again


def read_metrics(
    run_dir: str | Path, offset: int = 0
) -> tuple[List[Dict[str, Any]], int]:
    """Read metric records from ``offset`` bytes on.

    Returns ``(records, new_offset)`` so the dashboard can poll incrementally
    instead of re-reading a growing file on every request.
    """
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return [], 0

    records: List[Dict[str, Any]] = []
    with open(path) as fh:
        fh.seek(offset)
        for line in fh:
            if not line.endswith("\n"):
                break  # partial final line; pick it up next poll
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        new_offset = fh.tell()
    return records, new_offset


def list_runs(runs_dir: str | Path) -> List[Dict[str, Any]]:
    """Every run directory, newest first."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        manifest = read_manifest(directory) or {}
        out.append(
            {
                "name": directory.name,
                "path": str(directory),
                "status": manifest.get("status", "unknown"),
                "started": manifest.get("started", directory.stat().st_mtime),
                "model_parameters": manifest.get("model_parameters"),
                "best_val": manifest.get("best_val"),
                "has_checkpoint": (directory / "best.pt").exists(),
            }
        )
    return sorted(out, key=lambda r: r["started"] or 0, reverse=True)
