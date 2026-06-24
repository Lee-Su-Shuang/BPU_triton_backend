# BPU Triton Backend for RDK S100P

本目录是本次 **BPU Triton Backend** 开发任务的独立交付包。它可以脱离原开发工作区使用，用于向老师展示：如何在 RDK S100P 板子上开发 Triton Server 自定义 backend，通过 BPU 加速运行 `.hbm` 模型，并通过 HTTP 和网页实时画面完成端到端验证。

## 1. 任务目标

开发一个适配 RDK S100P BPU 的 Triton Server backend，使 Triton 可以通过自定义 `bpu` backend 加载并运行 Horizon/D-Robotics `.hbm` 模型。

本交付包完成的目标包括：

- 实现 Triton C++ custom backend：`libtriton_bpu.so`。
- 使用 Horizon/D-Robotics DNN/HBRT runtime 调用 S100P BPU 执行 `.hbm` 推理。
- 支持 YOLOv5x `.hbm` 模型的 Triton HTTP inference。
- 提供 HTTP e2e 测试脚本，验证 Triton -> backend -> BPU 链路。
- 提供 USB 摄像头实时网页验证链路，浏览器可以看到实时检测画面。
- 提供一键验证脚本，保证交付目录可复现。

## 2. 最终效果

实时演示链路：

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

浏览器访问：

```text
http://192.168.1.154:8080/
```

## 3. 快速验收

进入目录：

```bash
cd /home/sunrise/BPU_triton_backend
```

运行完整验证：

```bash
./verify_all.sh
```

验证脚本会自动完成：

1. 检查外部依赖。
2. 编译 C++ BPU backend。
3. 运行 C++ 单元测试。
4. 运行 Python realtime harness 测试。
5. 启动 Triton Server。
6. 执行 Triton HTTP e2e 推理。
7. 检查 USB 摄像头。
8. 临时启动网页服务并 smoke test `/health` 和 `/stream.mjpg`。
9. 停止服务。

成功时会输出：

```text
All BPU Triton backend deliverable checks passed.
```

## 4. 启动实时网页演示

```bash
cd /home/sunrise/BPU_triton_backend
./run_realtime_yolo.sh
```

然后浏览器打开：

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
  MANIFEST.md                文件清单、外部依赖、版本信息
  verify_all.sh              一键完整验收脚本
  run_realtime_yolo.sh       启动 Triton + 实时网页验证
  stop_services.sh           停止 Triton/web 服务

  bpu_backend/               C++ Triton BPU backend 源码和测试
  examples/yolov5x/          YOLOv5x Triton model repository 配置
  examples/realtime_yolo/    USB camera -> Triton HTTP -> BPU -> web 验证 harness
  scripts/triton_e2e/        Triton 启停和 HTTP e2e 脚本
  tests/                     交付包脚本测试
  docs/                      架构、实现、测试、演示文档
```

## 6. 推荐阅读顺序

给老师快速看：

```text
README.md
  -> docs/PROJECT_SUMMARY.md
  -> docs/DEMO_GUIDE.md
  -> docs/TESTING.md
```

深入看实现：

```text
docs/ARCHITECTURE.md
  -> docs/IMPLEMENTATION_NOTES.md
  -> bpu_backend/src/*.cc
  -> examples/realtime_yolo/*.py
```

## 7. 外部依赖

本目录不复制板载模型和 Triton Server 二进制包，依赖如下：

```text
YOLOv5x model:       /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
Triton Server dist:  /home/sunrise/triton_backend_BPU/tritonserver_dist
BPU runtime libs:    /usr/hobot/lib/libdnn.so, /usr/hobot/lib/libhbrt4.so
BPU devices:         /dev/bpu, /dev/bpu_core0
USB camera:          /dev/video0
```

详情见 `MANIFEST.md`。

## 8. 当前验证状态

本目录已经在当前 RDK S100P 板子上运行过 `./verify_all.sh`，验证通过。主要成功标准：

- C++ backend tests：`100% tests passed, 0 tests failed out of 7`
- Python/package tests：`17 passed`
- Triton HTTP e2e 输出：`7197120 / 1799280 / 449820`
- USB camera：可打开并读取 `(480, 640, 3)` BGR frame
- Web stream：`/stream.mjpg` 返回 multipart JPEG 数据
