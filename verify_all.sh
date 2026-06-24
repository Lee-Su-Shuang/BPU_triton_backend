#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FILE="${MODEL_FILE:-/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm}"
TRITON_DIST_DIR="${TRITON_DIST_DIR:-/home/sunrise/triton_backend_BPU/tritonserver_dist}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
PORT="${PORT:-8080}"

check_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

cd "$ROOT"
export TRITON_DIST_DIR

check_file "$MODEL_FILE" "YOLOv5x HBM model"
check_file "$TRITON_DIST_DIR/server/tritonserver/bin/tritonserver" "Triton Server binary"
check_file "/usr/hobot/lib/libdnn.so" "DNN runtime"
check_file "/usr/hobot/lib/libhbrt4.so" "HBRT runtime"
check_file "/dev/bpu" "BPU device"
check_file "$VIDEO_DEVICE" "USB camera device"

echo "[1/7] Build BPU backend"
cmake -S bpu_backend -B build/bpu_backend -DBPU_BACKEND_BUILD_TRITON=ON
cmake --build build/bpu_backend -j2

echo "[2/7] Run C++ backend tests"
ctest --test-dir build/bpu_backend --output-on-failure

echo "[3/7] Run realtime harness and package Python tests"
python3 -m pytest examples/realtime_yolo tests -v

echo "[4/7] Run Triton HTTP e2e"
./stop_services.sh >/dev/null 2>&1 || true
scripts/triton_e2e/start_bpu_triton.sh
python3 scripts/triton_e2e/infer_yolov5x_http.py

echo "[5/7] Check USB camera capture"
python3 - <<PY
import cv2
cap = cv2.VideoCapture('${VIDEO_DEVICE}')
print('camera_opened', cap.isOpened())
ok, frame = cap.read() if cap.isOpened() else (False, None)
print('camera_read', ok, 'shape', None if frame is None else frame.shape)
cap.release()
if not ok or frame is None:
    raise SystemExit(1)
PY

echo "[6/7] Start realtime web harness for endpoint smoke"
python3 examples/realtime_yolo/run_usb_yolov5x_web.py \
  --video-device "$VIDEO_DEVICE" \
  --triton-url http://127.0.0.1:8000 \
  --host 0.0.0.0 \
  --port "$PORT" > /tmp/bpu_triton_backend_realtime_verify.log 2>&1 &
WEB_PID=$!
cleanup() {
  kill "$WEB_PID" 2>/dev/null || true
  scripts/triton_e2e/stop_bpu_triton.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[7/7] Smoke web health and MJPEG stream"
python3 - <<PY
import time
import urllib.request
base = 'http://127.0.0.1:${PORT}'
last = None
for _ in range(40):
    try:
        data = urllib.request.urlopen(base + '/health', timeout=2).read().decode()
        print('health', data)
        break
    except Exception as exc:
        last = exc
        time.sleep(1)
else:
    raise RuntimeError(f'web health not ready: {last}')
stream = urllib.request.urlopen(base + '/stream.mjpg', timeout=30)
chunk = stream.read(4096)
stream.close()
print('stream_prefix', chunk[:80])
if b'--frame' not in chunk or b'\xff\xd8' not in chunk:
    raise RuntimeError('stream did not contain multipart JPEG data')
PY

echo "All BPU Triton backend deliverable checks passed."
