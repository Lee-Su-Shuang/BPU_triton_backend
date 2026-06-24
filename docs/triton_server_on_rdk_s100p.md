# Triton Server on RDK S100P

`tritonserver` is not installed in the board `PATH` by default:

```bash
command -v tritonserver
```

Initial result:

```text
tritonserver not found in PATH
```

## Validated standalone package

The official Triton Server GitHub release now provides an aarch64 standalone zip. This board was validated with:

```text
tritonserver-2.69.0+nv26.05-51748363-cu132-cp312-manylinux_2_28-aarch64.zip
```

Downloaded under:

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-2.69.0-aarch64.zip
```

Extracted under:

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver
```

Important caveat: this official package is an NVIDIA `+nv26.05-cu132` build and has dynamic dependencies on `libcudart.so.13` and `libdcgm.so.4`. RDK S100P is not an NVIDIA CUDA platform, so the repo includes a local no-GPU stub workflow for validation. The stubs are used only through `LD_LIBRARY_PATH` for this downloaded Triton Server process; they are not installed system-wide.

## Build local no-GPU stubs

From the repo root:

```bash
scripts/triton_e2e/build_no_gpu_stubs.sh /home/sunrise/triton_backend_BPU/tritonserver_dist
```

This produces:

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/stubs/libcudart.so.13
/home/sunrise/triton_backend_BPU/tritonserver_dist/stubs/libdcgm.so.4
```

Expected Triton startup warnings with these stubs:

```text
Unable to allocate pinned system memory, pinned memory pool will not be available: no CUDA-capable device is detected
CUDA memory pool disabled
CudaDriverHelper has not been initialized.
```

These warnings are acceptable for the BPU backend validation because the BPU backend uses CPU request buffers plus the Horizon/D-Robotics DNN/HBRT runtime, not CUDA.

## Start Triton with the BPU backend

From the repo root:

```bash
scripts/triton_e2e/start_bpu_triton.sh
```

The script:

1. Builds `build/bpu_backend/libtriton_bpu.so`.
2. Installs it into:

   ```text
   /home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver/backends/bpu/libtriton_bpu.so
   ```

3. Runs `examples/yolov5x/make_symlink.sh`.
4. Starts Triton Server with:

   ```text
   --backend-directory=/home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver/backends
   --model-repository=<repo>/examples/yolov5x/model_repository
   --http-port=8000
   --grpc-port=8001
   --metrics-port=8002
   ```

Runtime log:

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-bpu.log
```

PID file:

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-bpu.pid
```

## Validate HTTP inference

From the repo root:

```bash
scripts/triton_e2e/infer_yolov5x_http.py
```

Expected output sizes:

```text
raw_bytes 9446220 parsed_bytes 9446220 sizes {'output': 7197120, '1310': 1799280, '1312': 449820}
```

The validation run on 2026-06-23 successfully loaded the model:

```text
| Model       | Version | Status |
| yolov5x_bpu | 1       | READY  |
```

The metadata endpoint returned:

```text
inputs: data_y UINT8 [1,672,672,1], data_uv UINT8 [1,336,336,2]
outputs: output INT32 [1,84,84,255], 1310 INT32 [1,42,42,255], 1312 INT32 [1,21,21,255]
```

The deterministic HTTP inference request completed successfully with output byte sizes matching local `bpu_runner` validation.

## Stop Triton

```bash
scripts/triton_e2e/stop_bpu_triton.sh
```

## Remaining production consideration

This validates the BPU backend end-to-end through Triton HTTP on RDK S100P using the official aarch64 NVIDIA-linked standalone package plus local no-GPU stubs. For production packaging, prefer one of:

1. A CPU-only/non-CUDA Triton Server build for aarch64 Ubuntu 22.04.
2. A Triton Server build configured without NVIDIA GPU/DCGM dependencies.
3. A maintained deployment image that includes only the runtime dependencies required for the BPU backend.
