#!/usr/bin/env bash
# Run a collection campaign to completion, restarting CARLA when it dies.
#
# D51: a map load is minutes of heavy I/O, and on a slow disk it can take the
# simulator down with it (observed: SIGSEGV part-way through loading Town02,
# with one episode already banked). A full campaign needs one load per town, so
# the probability of surviving every load unattended is poor. The collector
# skips episodes already on disk, so relaunching resumes rather than restarts;
# this script supplies the relaunching.
#
# Usage: bash scripts/collect_supervised.sh configs/collect_pilot.yaml [max_restarts]
set -uo pipefail

CONFIG="${1:?usage: collect_supervised.sh <config.yaml> [max_restarts]}"
MAX_RESTARTS="${2:-20}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/Desktop/github_projects/3D_reconstruction/vendor/CARLA_0.9.15}"
CARLA_LOG="${CARLA_LOG:-/tmp/carla_supervised.log}"
SETTLE_SECONDS="${SETTLE_SECONDS:-60}"

# D54: force the discrete GPU on a hybrid-graphics laptop.
#
# This machine has an Intel UHD iGPU alongside an RTX 4050, and CARLA defaulted
# to the iGPU: nvidia-smi showed VRAM pinned at its 276 MiB idle baseline while
# six camera views were being rendered. The consequence was a render thread
# that collapsed as soon as the sensor load rose -- CARLA died attaching the
# *second* camera on Town02. With the discrete GPU it survives four to five,
# and VRAM use jumps to ~4.9 GB, which is what rendering actually costs.
#
# This very likely also explains the "GameThread timed out waiting for
# RenderThread" crashes early in this work, which were attributed to disk I/O.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] && \
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

carla_running() { ps -eo cmd | grep -q "[C]arlaUE4-Linux-Shipping"; }

start_carla() {
  # NOTE: no map argument. The packaged CARLA 0.9.15 build *ignores* it --
  # launching with `/Game/Carla/Maps/Town02` still comes up on Town10HD_Opt --
  # so the runtime `load_world` cannot be avoided by booting into the right
  # map. What a restart does buy is that the next load is the *first* in a
  # fresh process, unloading an empty default world rather than one populated
  # with sixty actors, which is the case that has survived here.
  local town
  town="$(python -u -m dav.collect --config "$CONFIG" --next-town 2>/dev/null | tail -1)"
  echo "[supervisor] starting CARLA${town:+ (next town: $town)}"
  # Keep a per-attempt log; overwriting it destroyed the crash evidence twice.
  CARLA_LOG_ATTEMPT="${CARLA_LOG%.log}_$(date +%H%M%S).log"
  ( cd "$CARLA_ROOT" && setsid nohup ./CarlaUE4.sh \
      -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000 \
      > "$CARLA_LOG_ATTEMPT" 2>&1 < /dev/null & )
  echo "[supervisor] carla log: $CARLA_LOG_ATTEMPT"
  # A cold map load off a rotational disk legitimately takes many minutes.
  for _ in $(seq 1 120); do
    sleep 10
    carla_running || { echo "[supervisor] CARLA died during boot"; return 1; }
    if python - <<'PY' 2>/dev/null
import carla, sys
c = carla.Client("127.0.0.1", 2000); c.set_timeout(20.0)
try:
    c.get_world().get_map()
except Exception:
    sys.exit(1)
PY
    then
      # Answering RPCs is not the same as being ready. The engine keeps
      # streaming assets for a while after the first reply, and spawning
      # sixty actors into that window is what blew the RPC deadline and
      # aborted the client (D52). Let it settle before handing over.
      echo "[supervisor] CARLA responding; settling for ${SETTLE_SECONDS}s"
      sleep "$SETTLE_SECONDS"
      return 0
    fi
  done
  echo "[supervisor] CARLA never became ready"
  return 1
}

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  carla_running || start_carla || { sleep 30; continue; }

  echo "[supervisor] collection attempt $attempt/$MAX_RESTARTS"
  python -u -m dav.collect --config "$CONFIG"
  status=$?

  if [ "$status" -eq 0 ]; then
    echo "[supervisor] collection finished"
    exit 0
  fi

  echo "[supervisor] collector exited $status; cleaning up before retry"
  # If the simulator is wedged rather than dead, replacing it is cheaper than
  # waiting: a hung UE4 process will not recover on its own.
  pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null
  pkill -9 -f "CarlaUE4.sh" 2>/dev/null
  sleep 15
done

echo "[supervisor] giving up after $MAX_RESTARTS attempts"
exit 1
