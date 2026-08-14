#!/usr/bin/env bash
# Launch the training dashboard.
#   bash scripts/run_gui.sh [port]
set -euo pipefail
PORT="${1:-8080}"
python -m gui.server --runs-dir runs --videos-dir outputs/videos \
  --data-dir "${DAV_DATA:-data/dav_pilot}" --port "$PORT"
