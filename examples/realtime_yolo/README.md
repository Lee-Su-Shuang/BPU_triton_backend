# Realtime YOLOv5x Triton BPU Backend Validation Harness

This example validates the Triton `bpu` backend with a realtime USB camera input and browser visualization. Inference always goes through Triton HTTP, Triton Server, `libtriton_bpu.so`, and the S100 BPU runtime.

## Architecture

- `triton_http.py` is a generic Triton HTTP binary tensor client.
- `frame_source.py` provides frame sources; `/dev/video0` USB camera is the first source.
- `yolov5x_adapter.py` contains YOLOv5x-specific input/output contracts, preprocessing, post-processing, and annotation.
- `web_stream.py` serves a generic MJPEG stream and health endpoint.
- `run_usb_yolov5x_web.py` wires the layers together.

## Start Triton

From the repository root:

```bash
scripts/triton_e2e/start_bpu_triton.sh
```

## Start validation harness

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

Health endpoint:

```text
http://192.168.1.154:8080/health
```

## Stop

Press Ctrl-C in the validation harness terminal, then stop Triton:

```bash
scripts/triton_e2e/stop_bpu_triton.sh
```

## Portability notes

The Triton client and web shell are model-agnostic. YOLOv5x preprocessing, post-processing, and S100 board paths are isolated in `yolov5x_adapter.py`. To validate another model, add a new model adapter with its own Triton contract and post-processing.

`hrt_model_exec model_info` is used by the YOLOv5x adapter to read output scale metadata required for Python-side YOLO decoding, but inference still goes through Triton HTTP and the `bpu` backend.
