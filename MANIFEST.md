# BPU Triton Backend 交付清单

## 交付目录

```text
/home/sunrise/BPU_triton_backend
```

本仓库是面向 RDK S100P Triton `bpu` backend 的独立交付包，同时包含 HTTP 实时 YOLO 验证工具链。

## 包含内容

```text
bpu_backend/                      Triton backend 源码、运行时和 C++ 测试
examples/yolov5x/                 YOLOv5x model repository 配置和输入准备脚本
examples/realtime_yolo/           实时 USB 摄像头 HTTP 验证工具链
scripts/triton_e2e/               Triton 启停和 HTTP 端到端验证脚本
tests/                            仓库脚本测试
docs/                             项目概览、架构、实现、测试、演示文档
README.md                         项目入口说明
MANIFEST.md                       本清单
verify_all.sh                     完整可复现验证脚本
run_realtime_yolo.sh              启动 Triton 与实时网页验证
stop_services.sh                  停止 Triton 与网页验证
```

## 文档索引

```text
docs/PROJECT_SUMMARY.md            项目完成内容和目标
docs/ARCHITECTURE.md              Triton -> bpu backend -> BPU 的整体架构
docs/IMPLEMENTATION_NOTES.md      关键源码和实现细节
docs/TESTING.md                   测试脚本、预期输出和验证含义
docs/DEMO_GUIDE.md                公开演示与操作说明
docs/bpu_backend_build.md         后端构建与验证说明
docs/triton_server_on_rdk_s100p.md RDK S100P 上的 Triton Server 部署说明
docs/backend_platform_support_matrix.md 平台支持矩阵说明
docs/realtime_validation_harness_design.md 实时验证工具链设计说明
```

## 外部依赖

以下依赖不随仓库一并提交，目标板上应已具备：

| 依赖 | 路径 | 说明 |
| --- | --- | --- |
| YOLOv5x HBM 模型 | `/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm` | 板端提供的模型，仓库通过符号链接引用。 |
| Triton Server 分发包 | `/home/sunrise/triton_backend_BPU/tritonserver_dist` | 用于本地验证的 Triton Server aarch64 独立包。 |
| DNN 运行时 | `/usr/hobot/lib/libdnn.so` | Horizon/D-Robotics DNN 运行时。 |
| HBRT 运行时 | `/usr/hobot/lib/libhbrt4.so` | Horizon/D-Robotics HBRT 运行时。 |
| BPU 设备 | `/dev/bpu`, `/dev/bpu_core0` | BPU 推理所需设备节点。 |
| USB 摄像头 | `/dev/video0` | 实时网页验证所需输入源。 |

## 已验证版本

```text
Triton Server: 2.69.0
DNN:           3.13.6
HBRT:          4.7.5
BPULib:        2.1.2
OS:            Ubuntu 22.04.5 LTS / RDK OS 4.0.2-Beta
Board:         D-Robotics RDK S100 V1P0
```

## 主要验收标准

运行 `./verify_all.sh` 后应满足：

- C++ 后端测试：`100% tests passed, 0 tests failed out of 7`
- Python 实时和脚本测试：`17 passed`
- Triton HTTP 输出尺寸：
  - `output`: `7197120`
  - `1310`: `1799280`
  - `1312`: `449820`
- USB 摄像头可打开并返回 BGR 帧。
- Web `/health` 返回 JSON。
- 网页 `/stream.mjpg` 返回 multipart JPEG 数据。

## 模型处理方式

仓库不直接拷贝 `.hbm` 模型文件，而是通过符号链接引用板端模型：

```text
examples/yolov5x/model_repository/yolov5x_bpu/1/model.hbm -> /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

这样可以保持交付包轻量，并与板端已验证模型保持一致。
