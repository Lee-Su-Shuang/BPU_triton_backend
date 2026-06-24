#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FILE="${MODEL_FILE:-/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm}"
TRITON_DIST_DIR="${TRITON_DIST_DIR:-/home/sunrise/triton_backend_BPU/tritonserver_dist}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
TRITON_URL="${TRITON_URL:-http://127.0.0.1:8000}"

check_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

check_file "$MODEL_FILE" "YOLOv5x HBM model"
check_file "$TRITON_DIST_DIR/server/tritonserver/bin/tritonserver" "Triton Server binary"
check_file "$VIDEO_DEVICE" "USB camera device"

cd "$ROOT"
export TRITON_DIST_DIR

./stop_services.sh >/dev/null 2>&1 || true
scripts/triton_e2e/start_bpu_triton.sh

exec python3 examples/realtime_yolo/run_usb_yolov5x_web.py \
  --video-device "$VIDEO_DEVICE" \
  --triton-url "$TRITON_URL" \
  --host "$HOST" \
  --port "$PORT"
