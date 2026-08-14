#!/usr/bin/env bash
# Collect a defensive-driving dataset. Starts CARLA if it is not already up.
#
#   bash scripts/collect_data.sh configs/collect_pilot.yaml
#   nohup bash scripts/collect_data.sh configs/collect_full.yaml > collect.log 2>&1 &
set -euo pipefail

CONFIG="${1:-configs/collect_pilot.yaml}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/Desktop/github_projects/3D_reconstruction/vendor/CARLA_0.9.15}"
PORT="${CARLA_PORT:-2000}"

if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
  echo "CARLA not found at $CARLA_ROOT (override with CARLA_ROOT=...)" >&2
  exit 1
fi

started_carla=0
if ! (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "starting CARLA on port $PORT ..."
  # -RenderOffScreen keeps the simulator off the desktop compositor, which
  # matters on a laptop GPU that also has to hold the training process.
  "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -carla-rpc-port=$PORT -quality-level=Epic &
  CARLA_PID=$!
  started_carla=1
  for _ in $(seq 1 60); do
    sleep 2
    (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  done
  echo "CARLA up (pid $CARLA_PID)"
fi

cleanup() {
  if [[ $started_carla -eq 1 ]]; then
    echo "stopping CARLA"
    kill "$CARLA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python -m dav.collect --config "$CONFIG"
