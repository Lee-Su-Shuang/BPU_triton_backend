# RDK S100P BPU 的 Triton 后端

本仓库提供一个面向 RDK S100P BPU 的 Triton Server 自定义后端实现。项目通过 Horizon/D-Robotics DNN/HBRT runtime 加载并运行 `.hbm` 模型，并提供本地推理、Triton HTTP 端到端验证以及 USB 摄像头实时网页验证链路。

## 1. 项目目标

项目目标是让 Triton Server 可以通过自定义 `bpu` backend 调用 RDK S100P 板载 BPU，运行 Horizon/D-Robotics `.hbm` 模型。

当前仓库包含：

- Triton C++ 自定义后端，编译产物为 `libtriton_bpu.so`。
- 基于 Horizon/D-Robotics DNN/HBRT runtime 的 BPU 推理封装。
- YOLOv5x `.hbm` 模型的 Triton 模型仓库配置。
- Triton HTTP binary tensor 端到端推理验证脚本。
- USB 摄像头实时网页验证链路。
- 一键构建、测试、推理和服务 smoke test 脚本。

## 2. 验证链路

实时验证链路如下：

```text
USB camera /dev/video0
  -> Python realtime validation harness
  -> Triton HTTP binary tensor request
  -> Triton Server
  -> custom bpu backend libtriton_bpu.so
  -> Horizon DNN / HBRT
  -> S100P BPU running yolov5x_672x672_nv12.hbm
  -> Triton raw output tensors
  -> YOLOv5x postprocess and annotation
  -> MJPEG browser stream
```

浏览器访问地址：

```text
http://192.168.1.154:8080/
```

## 3. 快速验证

进入仓库根目录：

```bash
cd /home/sunrise/BPU_triton_backend
```

运行完整验证：

```bash
./verify_all.sh
```

脚本会依次完成：

1. 检查外部依赖。
2. 编译 C++ BPU backend。
3. 运行 C++ 单元测试。
4. 运行 Python 实时验证和脚本测试。
5. 启动 Triton Server。
6. 执行 Triton HTTP 端到端推理。
7. 检查 USB 摄像头。
8. 临时启动网页服务，并验证 `/health` 与 `/stream.mjpg`。
9. 停止相关服务。

成功时输出：

```text
All BPU Triton backend deliverable checks passed.
```

## 4. 启动实时网页验证

```bash
cd /home/sunrise/BPU_triton_backend
./run_realtime_yolo.sh
```

然后在浏览器打开：

```text
http://192.168.1.154:8080/
```

停止服务：

```bash
./stop_services.sh
```

## 5. 目录结构

```text
BPU_triton_backend/
  README.md                  项目入口说明
  MANIFEST.md                文件清单、外部依赖和版本信息
  verify_all.sh              一键完整验证脚本
  run_realtime_yolo.sh       启动 Triton 与实时网页验证
  stop_services.sh           停止 Triton 和网页服务

  bpu_backend/               C++ Triton BPU backend 源码和测试
  examples/yolov5x/          YOLOv5x Triton model repository 配置
  examples/realtime_yolo/    USB camera -> Triton HTTP -> BPU -> web 验证工具
  scripts/triton_e2e/        Triton 启停和 HTTP 端到端脚本
  tests/                     仓库脚本测试
  docs/                      架构、实现、测试和部署文档
```

## 6. 推荐阅读顺序

快速了解项目：

```text
README.md
  -> docs/PROJECT_SUMMARY.md
  -> docs/TESTING.md
  -> docs/DEMO_GUIDE.md
```

深入了解实现：

```text
docs/ARCHITECTURE.md
  -> docs/IMPLEMENTATION_NOTES.md
  -> bpu_backend/src/*.cc
  -> examples/realtime_yolo/*.py
```

## 7. 外部依赖

本仓库不复制板载模型和 Triton Server 二进制包，运行时依赖如下：

```text
YOLOv5x model:       /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
Triton Server dist:  /home/sunrise/triton_backend_BPU/tritonserver_dist
BPU runtime libs:    /usr/hobot/lib/libdnn.so, /usr/hobot/lib/libhbrt4.so
BPU devices:         /dev/bpu, /dev/bpu_core0
USB camera:          /dev/video0
```

详情见 `MANIFEST.md`。

## 8. 当前验证状态

本仓库已在 RDK S100P 板端运行过 `./verify_all.sh`，验证项包括：

- C++ 后端测试：`100% tests passed, 0 tests failed out of 7`
- Python/package tests：`17 passed`
- Triton HTTP e2e 输出尺寸：`7197120 / 1799280 / 449820`
- USB camera：可打开并读取 `(480, 640, 3)` BGR frame
- 网页视频流：`/stream.mjpg` 返回 multipart JPEG 数据
