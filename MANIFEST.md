# BPU Triton Backend Deliverable Manifest

## Package path

```text
/home/sunrise/BPU_triton_backend
```

Purpose: standalone source deliverable for the RDK S100P Triton `bpu` backend plus HTTP realtime YOLO validation harness.

## Included source and scripts

```text
bpu_backend/                      Triton backend source, runner, and C++ tests
examples/yolov5x/                 YOLOv5x model repository config and input preparation
examples/realtime_yolo/           Realtime USB camera HTTP validation harness
scripts/triton_e2e/               Triton startup/shutdown and HTTP e2e validation
tests/                            Package script tests
docs/                             Project summary, architecture, implementation, testing, demo docs
README.md                         Project overview and teacher-facing entry point
MANIFEST.md                       This manifest
verify_all.sh                     Full reproducibility verification
run_realtime_yolo.sh              Starts Triton and realtime web harness
stop_services.sh                  Stops Triton and realtime web harness
```

## Documentation map

```text
docs/PROJECT_SUMMARY.md           What was built and why
docs/ARCHITECTURE.md              Triton -> bpu backend -> BPU architecture
docs/IMPLEMENTATION_NOTES.md      Important source files and implementation details
docs/TESTING.md                   Test scripts, expected outputs, and validation meaning
docs/DEMO_GUIDE.md                Short guide for teacher-facing live demo
docs/bpu_backend_build.md         Backend build notes
docs/triton_server_on_rdk_s100p.md Triton Server setup on RDK S100P
docs/backend_platform_support_matrix.md Platform support notes
docs/realtime_validation_harness_design.md Realtime validation harness design
```

## External dependencies not vendored

These are expected to exist on the target RDK S100P board:

| Dependency | Path | Notes |
| --- | --- | --- |
| YOLOv5x HBM model | `/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm` | Board-provided model; referenced by symlink in the Triton model repository. |
| Triton Server distribution | `/home/sunrise/triton_backend_BPU/tritonserver_dist` | Official Triton Server aarch64 standalone package with local no-GPU CUDA/DCGM stubs. Override with `TRITON_DIST_DIR=/path`. |
| DNN runtime | `/usr/hobot/lib/libdnn.so` | Horizon/D-Robotics DNN runtime. |
| HBRT runtime | `/usr/hobot/lib/libhbrt4.so` | Horizon/D-Robotics HBRT runtime. |
| BPU device | `/dev/bpu`, `/dev/bpu_core0` | Required for BPU inference. |
| USB camera | `/dev/video0` | Required for realtime browser validation. Override with `VIDEO_DEVICE=/dev/videoX`. |

## Verified versions observed on this board

```text
Triton Server: 2.69.0
DNN:           3.13.6
HBRT:          4.7.5
BPULib:        2.1.2
OS:            Ubuntu 22.04.5 LTS / RDK OS 4.0.2-Beta
Board:         D-Robotics RDK S100 V1P0
```

## Primary success criteria

`./verify_all.sh` should report:

- C++ backend tests: `100% tests passed, 0 tests failed out of 7`
- Python realtime/package tests: `17 passed`
- Triton HTTP inference output sizes:
  - `output`: `7197120`
  - `1310`: `1799280`
  - `1312`: `449820`
- USB camera opens and returns a BGR frame.
- Web `/health` returns JSON.
- Web `/stream.mjpg` returns multipart JPEG data.

## Model handling

The package does not copy the `.hbm` model. `examples/yolov5x/make_symlink.sh` creates the Triton model repository symlink:

```text
examples/yolov5x/model_repository/yolov5x_bpu/1/model.hbm -> /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

This keeps the deliverable lightweight and tied to the board-provided validated model.
