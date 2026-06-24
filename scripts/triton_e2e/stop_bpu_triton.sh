#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${TRITON_DIST_DIR:-/home/sunrise/triton_backend_BPU/tritonserver_dist}"
PID="$DIST_DIR/tritonserver-bpu.pid"
if [[ -f "$PID" ]]; then
  kill "$(cat "$PID")" 2>/dev/null || true
  rm -f "$PID"
fi
