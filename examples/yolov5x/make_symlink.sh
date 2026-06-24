#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SRC="${1:-/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm}"
MODEL_SRC="$(cd "$(dirname "$MODEL_SRC")" && pwd)/$(basename "$MODEL_SRC")"
MODEL_DIR="$ROOT_DIR/model_repository/yolov5x_bpu/1"
CONFIG_DIR="$ROOT_DIR/model_repository/yolov5x_bpu"

if [[ ! -f "$MODEL_SRC" ]]; then
  echo "model file not found: $MODEL_SRC" >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
cp "$ROOT_DIR/config.pbtxt" "$CONFIG_DIR/config.pbtxt"
ln -sfn "$MODEL_SRC" "$MODEL_DIR/model.hbm"

echo "model repository ready: $ROOT_DIR/model_repository"
