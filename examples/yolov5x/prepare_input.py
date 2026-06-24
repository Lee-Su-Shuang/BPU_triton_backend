#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def bgr_to_nv12(image_bgr: np.ndarray) -> bytes:
    height, width = image_bgr.shape[:2]
    yuv_i420 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV_I420)
    flat = yuv_i420.reshape(-1)
    y_size = height * width
    uv_size = y_size // 4
    y = flat[:y_size]
    u = flat[y_size : y_size + uv_size]
    v = flat[y_size + uv_size : y_size + 2 * uv_size]
    uv = np.empty((uv_size * 2,), dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    return y.tobytes() + uv.tobytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out-dir", default="tmp_inputs")
    parser.add_argument("--size", type=int, default=672)
    args = parser.parse_args()
    if args.size != 672:
        raise SystemExit("yolov5x_672x672_nv12.hbm requires --size 672")

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"failed to read image: {args.image}")

    resized = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    nv12 = bgr_to_nv12(resized)
    y_size = args.size * args.size
    uv_size = args.size * args.size // 2
    if len(nv12) != y_size + uv_size:
        raise SystemExit(f"unexpected NV12 size: {len(nv12)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_y.bin").write_bytes(nv12[:y_size])
    (out_dir / "data_uv.bin").write_bytes(nv12[y_size:])
    print(f"wrote {out_dir / 'data_y.bin'} bytes={y_size}")
    print(f"wrote {out_dir / 'data_uv.bin'} bytes={uv_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
