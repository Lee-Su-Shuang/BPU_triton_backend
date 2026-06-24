# 实现说明

## C++ Backend 实现

### `bpu_backend/src/triton_backend.cc`

这是 Triton custom backend 的入口文件，主要实现以下 Triton backend API：

- `TRITONBACKEND_ModelInitialize`
- `TRITONBACKEND_ModelFinalize`
- `TRITONBACKEND_ModelInstanceInitialize`
- `TRITONBACKEND_ModelInstanceFinalize`
- `TRITONBACKEND_ModelInstanceExecute`

执行流程：

1. Triton 加载模型时，backend 从 model repository 中找到 `.hbm`。
2. `BpuModel` 加载 `.hbm` 并读取输入 / 输出 metadata。
3. 每次请求进入 `ModelInstanceExecute`。
4. backend 收集 Triton 请求输入缓冲区。
5. 调用 `BpuRuntime::Infer`。
6. 把输出原始字节写入 Triton 响应输出缓冲区。
7. Triton 通过 HTTP / gRPC 返回给客户端。

代码中额外做了以下检查：

- 仅接受 CPU input buffer。
- 检查 input buffer size，避免 overflow。
- 检查 output buffer memory type。
- `ModelSetState` 失败时释放 state，避免泄漏。
- response output 创建失败时正确返回 Triton error。

### `bpu_backend/src/bpu_model.cc`

负责 `.hbm` 模型加载和 metadata 提取。

关键点：

- 使用 Horizon/D-Robotics DNN API 加载模型。
- 获取模型 handle。
- 读取 input / output tensor properties。
- 转换 dtype、shape、字节大小、stride 等信息。

### `bpu_backend/src/bpu_runtime.cc`

负责实际推理。

关键点：

- 根据 `BpuModel` 的 tensor metadata 准备输入输出 tensors。
- 把 Triton 请求中的原始字节拷贝到 DNN 输入 tensor。
- 调用 BPU runtime 同步推理。
- 把输出 tensor 原始字节拷贝回后端输出缓冲区。

### `bpu_backend/tools/bpu_runner.cc`

本地验证工具，不依赖 Triton。

用途：

- 打印 `.hbm` metadata。
- 使用 raw input 文件执行 BPU 推理。
- dump output 文件。

该工具有助于区分问题来自 BPU runtime 还是 Triton backend 集成层。

## Python 实时验证工具链

位置：

```text
examples/realtime_yolo/
```

### `triton_http.py`

通用 Triton HTTP binary tensor 客户端。

职责：

- 构造 Triton binary tensor request。
- 使用 `urllib.request` 发送 HTTP inference。
- 解析 Triton 响应头和原始输出字节。

该模块不包含 YOLO、BPU、摄像头或网页逻辑，因此可以复用于其他模型验证。

### `frame_source.py`

输入源模块。

当前实现：

- `UsbCameraSource('/dev/video0')`
- 使用 OpenCV `VideoCapture` 读取 BGR frame。
- 校验 frame 是 `HxWx3 uint8`。

### `yolov5x_adapter.py`

YOLOv5x 模型适配层。

职责：

- 定义 Triton model name：`yolov5x_bpu`。
- 定义输入 contract：`data_y`, `data_uv`。
- 定义输出 contract：`output`, `1310`, `1312`。
- 把 BGR frame resize 到 `672x672`。
- 转换为 NV12 split-plane 输入。
- 读取 `hrt_model_exec model_info` 中的输出 scale 元数据。
- 在 Python 侧完成 YOLO 解码、NMS 和标注。

注意：推理本身始终通过 Triton HTTP -> `bpu` backend -> BPU 完成。`hrt_model_exec model_info` 只用于读取后处理所需 scale metadata。

### `web_stream.py`

网页流模块。

职责：

- 提供 `/` 页面。
- 提供 `/health` JSON 状态。
- 提供 `/stream.mjpg` MJPEG 实时视频流。
- 使用异步 capture / inference / web pipeline 避免卡顿。

### `run_usb_yolov5x_web.py`

组合入口：

```text
UsbCameraSource
  + Yolov5xAdapter
  + TritonHttpClient
  + ValidationFramePipeline
  + MJPEG web server
```

## 为什么没有把后处理放进 backend

本阶段目标是验证 Triton backend 能通用地运行 `.hbm` 模型并返回 raw output tensors。YOLO 后处理属于模型适配逻辑，不属于通用 BPU backend 的核心职责。

因此当前设计是：

```text
backend: 通用 .hbm 推理和 raw tensor 返回
adapter: YOLOv5x-specific preprocess/postprocess
```

这样后续支持其他 `.hbm` 模型时，只需要增加新的 model adapter，而不需要修改 backend 核心。
