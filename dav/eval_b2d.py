"""Bench2Drive evaluation driver and result aggregator.

Two things live here:

``run``       launches the Bench2Drive / CARLA Leaderboard 2.0 harness over the
              220 official routes with ``DAVAgent`` as the entry point.  This
              needs the Bench2Drive repository checked out and a CARLA server
              running, and takes 20-40 hours on a single consumer GPU.

``aggregate`` parses the harness's JSON output into the flat summary the
              dashboard reads (``<run>/bench2drive.json``), scored against the
              agreed target.

The target is DriveTransformer-L's published Bench2Drive result plus 9 points:
DS 63.46 -> 72.46, SR 35.01 -> 44.01.  The absolute top of the leaderboard is
far higher (TFv6 at DS 95.28), and beating *that* by 9 points is arithmetically
impossible because Driving Score is capped at 100.  See DOCUMENTATION.md, the
"Benchmark target" section.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Published DriveTransformer-L result on Bench2Drive (Jia et al., 2025).
BASELINE = {"name": "DriveTransformer-L", "driving_score": 63.46, "success_rate": 35.01}
TARGET_MARGIN = 9.0
TARGET = {
    "driving_score": BASELINE["driving_score"] + TARGET_MARGIN,
    "success_rate": BASELINE["success_rate"] + TARGET_MARGIN,
}

#: Bench2Drive's five ability categories.
ABILITIES = ("Merging", "Overtaking", "EmergencyBrake", "GiveWay", "TrafficSign")


def write_agent_config(
    checkpoint: Path, out_path: Path, device: str = "cuda", **extra
) -> Path:
    import yaml

    payload = {"checkpoint": str(checkpoint.resolve()), "device": device, **extra}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(payload, fh)
    return out_path


def run(
    bench2drive_root: Path,
    carla_root: Path,
    checkpoint: Path,
    out_dir: Path,
    routes: Optional[Path] = None,
    port: int = 2000,
    traffic_manager_port: int = 8000,
    repetitions: int = 1,
    resume: bool = True,
) -> int:
    """Launch the leaderboard evaluator. Returns its exit code."""
    if not bench2drive_root.exists():
        raise FileNotFoundError(
            f"Bench2Drive not found at {bench2drive_root}.\n"
            "Clone it with:\n"
            "  git clone https://github.com/Thinklab-SJTU/Bench2Drive.git"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    agent_config = write_agent_config(checkpoint, out_dir / "agent_config.yaml")
    routes = routes or bench2drive_root / "leaderboard" / "data" / "bench2drive220.xml"
    results = out_dir / "leaderboard_result.json"

    env = os.environ.copy()
    extra_paths = [
        str(bench2drive_root / "leaderboard"),
        str(bench2drive_root / "scenario_runner"),
        str(carla_root / "PythonAPI" / "carla"),
        str(Path(__file__).resolve().parent.parent),
    ]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + [env.get("PYTHONPATH", "")])
    env["CARLA_ROOT"] = str(carla_root)

    command = [
        sys.executable,
        str(bench2drive_root / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"),
        f"--routes={routes}",
        f"--repetitions={repetitions}",
        "--track=SENSORS",
        f"--checkpoint={results}",
        f"--agent={Path(__file__).resolve().parent / 'agents' / 'dav_agent.py'}",
        f"--agent-config={agent_config}",
        f"--port={port}",
        f"--traffic-manager-port={traffic_manager_port}",
        f"--resume={int(resume)}",
    ]

    print("launching:\n  " + " \\\n  ".join(command), flush=True)
    process = subprocess.run(command, env=env)

    if results.exists():
        aggregate(results, out_dir / "bench2drive.json")
    return process.returncode


def aggregate(leaderboard_json: Path, out_path: Path) -> Dict[str, Any]:
    """Flatten the leaderboard output into the dashboard's schema."""
    with open(leaderboard_json) as fh:
        payload = json.load(fh)

    records = payload.get("_checkpoint", {}).get("records", [])
    routes: List[Dict[str, Any]] = []
    ability_hits: Dict[str, List[float]] = {a: [] for a in ABILITIES}

    for record in records:
        scores = record.get("scores", {})
        driving_score = float(scores.get("score_composed", 0.0))
        route_completion = float(scores.get("score_route", 0.0))
        infraction_penalty = float(scores.get("score_penalty", 0.0))
        # The leaderboard marks a route successful when it completes with no
        # blocking infraction; Bench2Drive's SR uses the same definition.
        success = record.get("status", "") == "Completed" and route_completion >= 99.5

        name = record.get("route_id", "") or record.get("index", "")
        routes.append(
            {
                "name": str(name),
                "driving_score": driving_score,
                "route_completion": route_completion,
                "infraction_penalty": infraction_penalty,
                "success": bool(success),
                "status": record.get("status", ""),
                "infractions": {
                    k: len(v) if isinstance(v, list) else v
                    for k, v in record.get("infractions", {}).items()
                },
            }
        )

        # Bench2Drive encodes the ability in the route identifier.
        for ability in ABILITIES:
            if ability.lower() in str(name).lower():
                ability_hits[ability].append(100.0 if success else 0.0)

    total = len(routes)
    driving_score = sum(r["driving_score"] for r in routes) / total if total else 0.0
    success_rate = 100.0 * sum(r["success"] for r in routes) / total if total else 0.0

    summary = {
        "routes_total": total,
        "routes_completed": sum(1 for r in routes if r["status"] == "Completed"),
        "driving_score": driving_score,
        "success_rate": success_rate,
        "baseline": BASELINE,
        "target": TARGET,
        "target_met": {
            "driving_score": driving_score >= TARGET["driving_score"],
            "success_rate": success_rate >= TARGET["success_rate"],
        },
        "delta_vs_baseline": {
            "driving_score": driving_score - BASELINE["driving_score"],
            "success_rate": success_rate - BASELINE["success_rate"],
        },
        "abilities": {
            a: (sum(v) / len(v) if v else 0.0) for a, v in ability_hits.items()
        },
        "routes": routes,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Driving Score {driving_score:.2f} (target {TARGET['driving_score']:.2f}, "
        f"{'MET' if summary['target_met']['driving_score'] else 'not met'})\n"
        f"Success Rate  {success_rate:.2f} (target {TARGET['success_rate']:.2f}, "
        f"{'MET' if summary['target_met']['success_rate'] else 'not met'})\n"
        f"wrote {out_path}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Bench2Drive evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="launch the closed-loop evaluation")
    run_parser.add_argument("--bench2drive", required=True, help="Bench2Drive repo root")
    run_parser.add_argument("--carla", required=True, help="CARLA 0.9.15 root")
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--out-dir", required=True, help="usually the run directory")
    run_parser.add_argument("--routes", default=None)
    run_parser.add_argument("--port", type=int, default=2000)
    run_parser.add_argument("--traffic-manager-port", type=int, default=8000)
    run_parser.add_argument("--repetitions", type=int, default=1)

    agg_parser = sub.add_parser("aggregate", help="summarise an existing result file")
    agg_parser.add_argument("--input", required=True)
    agg_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "run":
        return run(
            Path(args.bench2drive), Path(args.carla), Path(args.checkpoint),
            Path(args.out_dir), Path(args.routes) if args.routes else None,
            args.port, args.traffic_manager_port, args.repetitions,
        )
    aggregate(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
