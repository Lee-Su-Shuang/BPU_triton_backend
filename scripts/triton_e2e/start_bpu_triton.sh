#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${TRITON_DIST_DIR:-/home/sunrise/triton_backend_BPU/tritonserver_dist}"
TRITON_ROOT="$DIST_DIR/server/tritonserver"
STUB_DIR="$DIST_DIR/stubs"
MODEL_REPO="$REPO_ROOT/examples/yolov5x/model_repository"
LOG="$DIST_DIR/tritonserver-bpu.log"
PID="$DIST_DIR/tritonserver-bpu.pid"

if [[ ! -x "$TRITON_ROOT/bin/tritonserver" ]]; then
  chmod +x "$TRITON_ROOT/bin/tritonserver"
fi

if [[ ! -f "$STUB_DIR/libcudart.so.13" || ! -f "$STUB_DIR/libdcgm.so.4" ]]; then
  "$REPO_ROOT/scripts/triton_e2e/build_no_gpu_stubs.sh" "$DIST_DIR"
fi

cmake -S "$REPO_ROOT/bpu_backend" -B "$REPO_ROOT/build/bpu_backend" -DBPU_BACKEND_BUILD_TRITON=ON
cmake --build "$REPO_ROOT/build/bpu_backend" -j2

mkdir -p "$TRITON_ROOT/backends/bpu"
cp "$REPO_ROOT/build/bpu_backend/libtriton_bpu.so" "$TRITON_ROOT/backends/bpu/libtriton_bpu.so"
"$REPO_ROOT/examples/yolov5x/make_symlink.sh"

if [[ -f "$PID" ]]; then
  kill "$(cat "$PID")" 2>/dev/null || true
  rm -f "$PID"
fi
rm -f "$LOG"

LD_LIBRARY_PATH="$STUB_DIR:$TRITON_ROOT/lib64:/usr/hobot/lib:${LD_LIBRARY_PATH:-}" \
"$TRITON_ROOT/bin/tritonserver" \
  --backend-directory="$TRITON_ROOT/backends" \
  --model-repository="$MODEL_REPO" \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --strict-readiness=true \
  --log-info=true \
  >"$LOG" 2>&1 &
echo $! > "$PID"
echo "tritonserver pid=$(cat "$PID") log=$LOG"
