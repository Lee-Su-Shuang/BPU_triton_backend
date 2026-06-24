# Project Summary

## 任务名称

开发适配 RDK S100P BPU 的 Triton Server backend，并完成 HTTP 与实时网页端到端验证。

## 完成内容概览

本项目实现了一个 Triton C++ custom backend，backend 名称为 `bpu`，编译产物为：

```text
libtriton_bpu.so
```

该 backend 能够加载 Horizon/D-Robotics `.hbm` 模型，通过板载 DNN/HBRT runtime 调用 S100P BPU 执行推理，并通过 Triton 标准 HTTP inference API 对外提供服务。

同时，本项目提供了一个实时验证 harness：通过 USB 摄像头采集画面，将画面预处理为 YOLOv5x `.hbm` 模型需要的 NV12 split-plane 输入，通过 Triton HTTP 调用 `yolov5x_bpu` 模型，再把检测结果绘制到网页 MJPEG 实时画面中。

## 已实现模块

### 1. C++ BPU backend

位置：

```text
bpu_backend/
```

核心能力：

- 加载 `.hbm` 模型。
- 读取输入/输出 tensor metadata。
- 封装 BPU runtime 同步推理。
- 实现 Triton backend 生命周期接口。
- 支持 Triton request input collection 和 response output 写回。
- 编译生成 `libtriton_bpu.so`。

### 2. 本地 BPU runner

位置：

```text
bpu_backend/tools/bpu_runner.cc
```

用途：

- 不经过 Triton，直接加载 `.hbm` 模型。
- 打印模型 metadata。
- 使用 raw input 文件执行本地 BPU 推理。
- 方便定位问题是 backend/Triton 层还是 BPU runtime 层。

### 3. YOLOv5x Triton model repository

位置：

```text
examples/yolov5x/
```

模型配置：

```text
model name: yolov5x_bpu
backend:    bpu
inputs:     data_y, data_uv
outputs:    output, 1310, 1312
```

`.hbm` 模型作为外部依赖：

```text
/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

### 4. Triton HTTP e2e scripts

位置：

```text
scripts/triton_e2e/
```

包含：

- `start_bpu_triton.sh`：编译 backend、安装 `libtriton_bpu.so`、准备 model repository、启动 Triton。
- `infer_yolov5x_http.py`：构造 Triton binary tensor HTTP request，验证 YOLOv5x raw output tensors。
- `stop_bpu_triton.sh`：停止 Triton。
- `build_no_gpu_stubs.sh`：为当前无 NVIDIA GPU 的板端环境准备 Triton Server 需要的 CUDA/DCGM stub。

### 5. Realtime YOLO validation harness

位置：

```text
examples/realtime_yolo/
```

功能：

- 使用 `/dev/video0` 采集 USB 摄像头画面。
- 转换 BGR frame 为 YOLOv5x 需要的 NV12 split-plane 输入。
- 使用 Triton HTTP binary tensor API 调用 `yolov5x_bpu`。
- 通过 Python YOLOv5x adapter 解析输出和绘制检测框。
- 提供 `/health` 和 `/stream.mjpg`。
- 使用异步采集/推理/显示 pipeline，避免网页画面被推理延迟阻塞。

## 验证结果

通过 `./verify_all.sh` 完成完整验证：

```text
C++ tests:       7/7 passed
Python tests:    17 passed
Triton HTTP e2e: output sizes 7197120 / 1799280 / 449820
Camera smoke:    /dev/video0 opened and read frame (480, 640, 3)
Web smoke:       /stream.mjpg returned multipart JPEG data
```

## 任务完成结论

本项目已完成老师布置的 BPU Triton backend 开发任务：

```text
Triton Server
  -> custom bpu backend
  -> Horizon DNN/HBRT
  -> RDK S100P BPU
  -> .hbm model inference
  -> HTTP API and realtime browser validation
```
