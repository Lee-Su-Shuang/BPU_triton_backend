# Realtime YOLOv5x Triton BPU Backend Validation Harness Design

Date: 2026-06-24

## Goal

Provide a portable realtime validation harness for the Triton `bpu` backend. The first visible use case is a browser page showing USB-camera YOLOv5x detections, but the architecture must validate and exercise the backend path rather than become a one-off board demo.

The validated path is:

```text
frame source
  -> model adapter preprocess
  -> Triton HTTP binary tensor request
  -> Triton Server
  -> bpu backend
  -> S100 BPU `.hbm` inference
  -> Triton HTTP binary tensor response
  -> model adapter postprocess
  -> optional visualization
```

## Current verified context

- Triton `bpu` backend builds and passes the local test suite.
- Triton HTTP inference for `yolov5x_bpu` has been verified end-to-end with output tensor sizes `7197120`, `1799280`, and `449820`.
- A USB camera is attached and OpenCV can read frames from `/dev/video0` with shape `(480, 640, 3)`.
- Python modules available on the board include `cv2`, `numpy`, `websockets`, and `PIL`; Flask/FastAPI/tritonclient are not available.
- `hrt_model_exec model_info` can export YOLOv5 output scale metadata for Python-side post-processing without loading a second inference runtime in the web process.

## Design principles

1. **Backend validation first**: inference must go through Triton HTTP and the `bpu` backend. The harness must not bypass Triton for inference.
2. **Keep backend generic**: do not put YOLOv5x, USB camera, or web display assumptions into backend code.
3. **Separate reusable layers**: Triton transport, frame sources, model adapters, and visualization should be independently testable.
4. **Make board-specific pieces explicit**: S100 paths, model metadata loading, and `/usr/lib/libpostprocess.so` belong in the YOLOv5x adapter or documentation, not in generic client code.
5. **Prefer installed dependencies**: use Python standard library, `cv2`, `numpy`, and board-provided libraries. Do not add Flask, FastAPI, or `tritonclient`.

## Proposed file structure

```text
examples/realtime_yolo/
  __init__.py
  triton_http.py          # generic Triton HTTP binary tensor client
  frame_source.py         # camera/file frame sources; USB camera first
  web_stream.py           # generic MJPEG web server and health endpoint
  yolov5x_adapter.py      # YOLOv5x-specific contract, preprocess, postprocess, annotation
  run_usb_yolov5x_web.py  # thin composition entry point
  README.md               # operator and validation instructions
```

## Layer responsibilities

### Generic Triton HTTP client

`triton_http.py` knows how to:

- Build Triton binary tensor inference requests from model name, inputs, and requested outputs.
- Send HTTP requests with `urllib.request`.
- Parse Triton binary tensor responses into named raw output buffers.

It must not know about YOLO, NV12, cameras, BPU, or web rendering.

### Frame source layer

`frame_source.py` provides input adapters. The first adapter is USB camera capture through OpenCV:

- Default device: `/dev/video0`.
- Output frame format: BGR `numpy.ndarray`.

The interface should allow later sources such as video files, RTSP, or image directories without changing Triton client or model adapter code.

### YOLOv5x model adapter

`yolov5x_adapter.py` owns all YOLOv5x-specific knowledge:

- Triton model name: `yolov5x_bpu`.
- Input contract:
  - `data_y`: shape `[1, 672, 672, 1]`, datatype `UINT8`.
  - `data_uv`: shape `[1, 336, 336, 2]`, datatype `UINT8`.
- Output contract:
  - `output`: shape `[1, 84, 84, 255]`, datatype `INT32`.
  - `1310`: shape `[1, 42, 42, 255]`, datatype `INT32`.
  - `1312`: shape `[1, 21, 21, 255]`, datatype `INT32`.
- BGR to NV12 split-plane preprocessing.
- Python-side YOLO decoding using output scale metadata from `hrt_model_exec model_info`.
- Annotation drawing.

If post-processing requires metadata that Triton does not expose in the current backend response, the adapter may load the same `.hbm` model using board metadata APIs strictly for output tensor metadata. That metadata path must be documented as post-processing support only; inference remains Triton-only.

### Web visualization shell

`web_stream.py` exposes:

- `/` HTML page.
- `/stream.mjpg` MJPEG stream.
- `/health` JSON status.

It should consume a frame generator and know nothing about YOLO details except receiving already annotated JPEG frames or status messages.

### Thin USB YOLO web entry point

`run_usb_yolov5x_web.py` wires the layers together:

```text
UsbCameraSource -> Yolov5xAdapter -> TritonHttpClient -> MjpegServer
```

It should contain argument parsing and lifecycle management only.

## Runtime expectations

Start Triton separately:

```bash
scripts/triton_e2e/start_bpu_triton.sh
```

Start the validation harness:

```bash
python3 examples/realtime_yolo/run_usb_yolov5x_web.py \
  --video-device /dev/video0 \
  --triton-url http://127.0.0.1:8000 \
  --host 0.0.0.0 \
  --port 8080
```

Open:

```text
http://192.168.1.154:8080/
```

## Error handling

- If the camera cannot be opened, fail fast with the selected device path.
- If Triton is unreachable or returns an invalid response, keep the web server alive and expose the error through `/health` and frame overlay.
- If post-processing fails, stream camera frames with an error overlay so the backend transport path can still be debugged.
- On Ctrl-C, release camera resources cleanly.

## Testing and verification

Implementation should be verified with:

1. Unit tests for generic Triton request/response handling.
2. Unit tests for frame source preprocessing and JPEG/web helpers.
3. Existing backend build and tests:

   ```bash
   cmake -S bpu_backend -B build/bpu_backend -DBPU_BACKEND_BUILD_TRITON=ON
   cmake --build build/bpu_backend -j2
   ctest --test-dir build/bpu_backend --output-on-failure
   ```

4. Existing Triton HTTP e2e script:

   ```bash
   scripts/triton_e2e/start_bpu_triton.sh
   python3 scripts/triton_e2e/infer_yolov5x_http.py
   scripts/triton_e2e/stop_bpu_triton.sh
   ```

5. USB camera one-frame validation through Triton:

   - Capture `/dev/video0` frame.
   - Preprocess with `Yolov5xAdapter`.
   - Infer through `TritonHttpClient`.
   - Verify raw output sizes.

6. Web endpoint smoke test:

   - Start the harness.
   - Fetch `/health`.
   - Fetch the first multipart chunk from `/stream.mjpg`.
   - Manually open the browser page and confirm live annotated frames.

## Out of scope for this first validation harness

- Modifying the Triton backend to perform YOLO post-processing internally.
- Adding WebSocket/canvas frontend architecture.
- Supporting multiple simultaneous cameras.
- Supporting arbitrary models without model-specific adapters.
- Production authentication, TLS, or external publishing.
- Containerizing the full harness.
