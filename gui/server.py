"""FastAPI server for the DAV training dashboard.

    python -m gui.server --runs-dir runs --port 8080

Then open http://127.0.0.1:8080. Over SSH:
    ssh -L 8080:127.0.0.1:8080 <host>

The dashboard polls incrementally: it remembers a byte offset into each run's
``metrics.jsonl`` and asks only for what is new, so tailing a run that has been
training for days costs the same per poll as one that just started.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dav.utils.logging import list_runs, read_manifest, read_metrics

HERE = Path(__file__).resolve().parent


def create_app(runs_dir: Path, videos_dir: Path, data_dir: Optional[Path]) -> FastAPI:
    app = FastAPI(title="DAV Training Dashboard")

    def run_path(name: str) -> Path:
        # Reject anything that escapes the runs directory. The name arrives
        # from a URL, so ".." would otherwise read arbitrary files.
        candidate = (runs_dir / name).resolve()
        if not str(candidate).startswith(str(runs_dir.resolve())):
            raise HTTPException(400, "invalid run name")
        if not candidate.is_dir():
            raise HTTPException(404, f"no such run: {name}")
        return candidate

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (HERE / "static" / "index.html").read_text()

    @app.get("/api/runs")
    def api_runs() -> List[Dict[str, Any]]:
        return list_runs(runs_dir)

    @app.get("/api/runs/{name}/manifest")
    def api_manifest(name: str) -> Dict[str, Any]:
        manifest = read_manifest(run_path(name))
        if manifest is None:
            raise HTTPException(404, "no manifest yet")
        return manifest

    @app.get("/api/runs/{name}/metrics")
    def api_metrics(name: str, offset: int = Query(0, ge=0)) -> Dict[str, Any]:
        records, new_offset = read_metrics(run_path(name), offset)
        return {"records": records, "offset": new_offset}

    @app.get("/api/runs/{name}/eval")
    def api_eval(name: str) -> Dict[str, Any]:
        path = run_path(name) / "bench2drive.json"
        if not path.exists():
            return {"available": False}
        with open(path) as fh:
            return {"available": True, **json.load(fh)}

    @app.get("/api/videos")
    def api_videos() -> List[Dict[str, Any]]:
        if not videos_dir.exists():
            return []
        out = []
        for path in sorted(videos_dir.glob("*.mp4")):
            sidecar = path.with_suffix(".json")
            meta = {}
            if sidecar.exists():
                try:
                    with open(sidecar) as fh:
                        meta = json.load(fh)
                except json.JSONDecodeError:
                    pass
            out.append(
                {
                    "name": path.name,
                    "url": f"/videos/{path.name}",
                    "size_mb": round(path.stat().st_size / 1e6, 1),
                    "modified": path.stat().st_mtime,
                    "meta": meta,
                }
            )
        return sorted(out, key=lambda v: v["modified"], reverse=True)

    @app.get("/videos/{filename}")
    def video(filename: str):
        path = (videos_dir / filename).resolve()
        if not str(path).startswith(str(videos_dir.resolve())) or not path.exists():
            raise HTTPException(404, "no such video")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/dataset")
    def api_dataset() -> Dict[str, Any]:
        """Compliance summary of the collected dataset, if one is present."""
        if data_dir is None or not data_dir.exists():
            return {"available": False}
        manifest_path = data_dir / "manifest.json"
        episodes_dir = data_dir / "episodes"
        if not episodes_dir.exists():
            return {"available": False}

        episodes = []
        for path in sorted(episodes_dir.glob("*.json")):
            try:
                with open(path) as fh:
                    episodes.append(json.load(fh))
            except json.JSONDecodeError:
                continue

        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path) as fh:
                    manifest = json.load(fh)
            except json.JSONDecodeError:
                pass

        return {
            "available": True,
            "root": str(data_dir),
            "episodes": episodes,
            "accepted": manifest.get("accepted_episodes", len(episodes)),
            "rejected": manifest.get("rejected_attempts", 0),
            "total_frames": manifest.get(
                "total_frames", sum(e.get("frames", 0) for e in episodes)
            ),
        }

    static_dir = HERE / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="DAV training dashboard")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--videos-dir", default="outputs/videos")
    parser.add_argument("--data-dir", default=None, help="dataset root for the data panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        Path(args.runs_dir),
        Path(args.videos_dir),
        Path(args.data_dir) if args.data_dir else None,
    )
    print(f"dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
