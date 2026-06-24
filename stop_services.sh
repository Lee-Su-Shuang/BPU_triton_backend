#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_DIST_DIR="${TRITON_DIST_DIR:-/home/sunrise/triton_backend_BPU/tritonserver_dist}"
PORT="${PORT:-8080}"

cd "$ROOT"
export TRITON_DIST_DIR

if command -v pgrep >/dev/null 2>&1; then
  pkill -f "examples/realtime_yolo/run_usb_yolov5x_web.py.*--port ${PORT}" 2>/dev/null || true
fi

scripts/triton_e2e/stop_bpu_triton.sh >/dev/null 2>&1 || true

# Clean up any leftover Triton process using the standard ports from this package.
for pid in $(pgrep -f "tritonserver.*--http-port=8000.*--grpc-port=8001" 2>/dev/null || true); do
  kill "$pid" 2>/dev/null || true
done
sleep 1
for pid in $(pgrep -f "tritonserver.*--http-port=8000.*--grpc-port=8001" 2>/dev/null || true); do
  kill -9 "$pid" 2>/dev/null || true
done

echo "Stopped realtime validation services."
