#!/usr/bin/env bash
# Train the safety-aware DriveTransformer.
#   bash scripts/train.sh configs/dav_tiny.yaml
set -euo pipefail
CONFIG="${1:-configs/dav_tiny.yaml}"
shift || true
python -m dav.train --config "$CONFIG" "$@"
