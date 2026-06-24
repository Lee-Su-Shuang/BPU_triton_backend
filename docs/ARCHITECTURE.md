# 系统架构

## 总体架构

本项目的核心目标是让 Triton Server 能够通过自定义 `bpu` backend 调用 RDK S100P 板载 BPU 执行 `.hbm` 模型推理。

整体架构：

```text
Client / HTTP request
  -> Triton Server
  -> bpu backend: libtriton_bpu.so
  -> BpuModel / BpuRuntime / BpuTensor
  -> Horizon DNN API / HBRT
  -> RDK S100P BPU
  -> .hbm model
  -> Triton response
```

## Backend 内部结构

```text
bpu_backend/
  include/bpu_backend/
    bpu_status.h        Result/Status 错误处理
    bpu_tensor.h        tensor metadata 和 raw buffer 描述
    bpu_model.h         .hbm 模型加载和 metadata 读取
    bpu_runtime.h       BPU 推理封装
    bpu_runner_args.h   CLI runner 参数解析

  src/
    bpu_tensor.cc
    bpu_model.cc
    bpu_runtime.cc
    bpu_runner_args.cc
    triton_backend.cc   Triton backend API 实现

  tools/
    bpu_runner.cc       本地调试工具

  tests/
    C++ 单元测试
```

### BpuModel

职责：

- 使用 DNN/HBRT runtime 加载 `.hbm` 模型。
- 获取模型名称。
- 获取输入 tensor metadata。
- 获取输出 tensor metadata。
- 把 DNN tensor 属性转换为后端内部 `TensorMeta`。

### BpuRuntime

职责：

- 接收输入 raw buffers。
- 分配和准备 DNN tensor。
- 调用 BPU runtime 执行同步推理。
- 收集输出 raw buffers。

### triton_backend.cc

职责：

- 实现 Triton backend lifecycle。
- 在 ModelInitialize 阶段加载 `.hbm`。
- 在 ModelInstanceExecute 阶段处理 Triton requests。
- 从 Triton 请求中收集 CPU 输入缓冲区。
- 调用 `BpuRuntime::Infer`。
- 创建 Triton response outputs。
- 写回 raw output tensors。

## Triton 模型仓库

YOLOv5x 模型仓库：

```text
examples/yolov5x/model_repository/yolov5x_bpu/
  config.pbtxt
  1/model.hbm -> /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

`config.pbtxt` 声明：

```text
name: "yolov5x_bpu"
backend: "bpu"
input:  data_y, data_uv
output: output, 1310, 1312
```

## HTTP 端到端验证链路

```text
infer_yolov5x_http.py
  -> build Triton binary tensor HTTP request
  -> POST /v2/models/yolov5x_bpu/infer
  -> Triton Server
  -> bpu backend
  -> BPU inference
  -> Triton binary tensor response
  -> verify output byte sizes
```

预期输出尺寸：

```text
output: 7197120
1310:   1799280
1312:   449820
```

## 实时网页验证架构

```text
USB camera /dev/video0
  -> frame_source.py
  -> yolov5x_adapter.py preprocess: BGR -> NV12 data_y/data_uv
  -> triton_http.py
  -> Triton Server / bpu backend / BPU
  -> yolov5x_adapter.py postprocess and annotate
  -> web_stream.py MJPEG stream
  -> browser
```

为了降低网页卡顿，实时验证工具链采用异步 pipeline：

```text
Capture thread:
  持续读取 USB 最新帧

Inference thread:
  使用最新帧调用 Triton/BPU，更新最新检测结果

Web stream thread:
  按目标 FPS 输出最新画面 + 最新检测框
```

这样视频显示不会被单次推理延迟阻塞。检测框按 BPU/Triton 推理速度更新，画面按网页流速度更新。

## 外部依赖边界

本项目源码不复制以下依赖：

- Triton Server standalone package
- YOLOv5x `.hbm` 模型
- Hobot DNN/HBRT runtime libraries
- BPU device nodes

这些依赖由 RDK S100P 板端系统提供，详见 `MANIFEST.md`。
