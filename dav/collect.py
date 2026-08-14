"""CLI for dataset collection.

    python -m dav.collect --config configs/collect_pilot.yaml

Requires a running CARLA server; ``scripts/collect_data.sh`` starts one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields

import yaml

from .data.collector import CollectConfig, collect, next_town


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect a defensive-driving dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="key=value overrides, e.g. episodes=2 frames_per_episode=200",
    )
    parser.add_argument(
        "--next-town", action="store_true",
        help=(
            "print the town of the next uncollected episode and exit. Lets a "
            "supervisor boot CARLA straight into that map instead of switching "
            "at runtime, which is where the simulator crashes on slow disks "
            "(D51). Prints nothing when the campaign is already complete."
        ),
    )
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        payload = yaml.safe_load(fh) or {}
    for override in args.overrides:
        key, _, value = override.partition("=")
        payload[key] = yaml.safe_load(value)

    known = {f.name for f in fields(CollectConfig)}
    unknown = set(payload) - known
    if unknown:
        # Silently ignoring a mistyped key would produce a dataset that quietly
        # differs from what the config says.
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")

    cfg = CollectConfig(**payload)

    if args.next_town:
        town = next_town(cfg)
        if town:
            print(town)
        return 0

    collect(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
