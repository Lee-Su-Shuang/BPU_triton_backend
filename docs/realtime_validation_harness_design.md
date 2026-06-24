# 实时 YOLOv5x Triton BPU 后端验证工具链设计

日期：2026-06-24

## 目标

提供一个可移植的 Triton `bpu` 后端实时验证工具链。首个可见场景是通过浏览器展示 USB 摄像头输入下的 YOLOv5x 检测结果，但工具链的核心目标是验证和覆盖后端链路，而不是做成一次性的板端演示程序。

验证链路如下：

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

## 当前已验证上下文

- Triton `bpu` backend 可编译并通过本地测试。
- `yolov5x_bpu` 的 Triton HTTP 推理已完成端到端验证，输出 tensor 尺寸分别为 `7197120`、`1799280`、`449820`。
- 目标板已连接 USB 摄像头，OpenCV 可以从 `/dev/video0` 读取形状为 `(480, 640, 3)` 的帧。
- 板端可用的 Python 模块包括 `cv2`、`numpy`、`websockets` 和 `PIL`；当前未使用 Flask、FastAPI 或 `tritonclient`。
- `hrt_model_exec model_info` 可导出 YOLOv5 输出 scale 元数据，供 Python 侧后处理使用，而无需在网页进程中再加载一套推理运行时。

## 设计原则

1. **先验证后端**：推理必须经过 Triton HTTP 和 `bpu` backend，不能在工具链里绕过 Triton。
2. **后端保持通用**：不要把 YOLOv5x、USB 摄像头或网页显示假设写入后端核心代码。
3. **分层可复用**：Triton 传输、帧源、模型适配器和可视化应该可以独立测试。
4. **显式标出板端依赖**：S100 路径、模型元数据加载和后处理相关依赖应放在 YOLOv5x 适配层或文档中，而不是写进通用客户端。
5. **优先使用已安装依赖**：使用 Python 标准库、`cv2`、`numpy` 和板端已有库，不额外引入 Flask、FastAPI 或 `tritonclient`。

## 建议的文件结构

```text
examples/realtime_yolo/
  __init__.py
  triton_http.py          # 通用 Triton HTTP binary tensor 客户端
  frame_source.py         # 摄像头或文件帧源；优先 USB 摄像头
  web_stream.py           # 通用 MJPEG 网页服务与健康检查接口
  yolov5x_adapter.py      # YOLOv5x 专属输入输出契约、预处理、后处理、标注
  run_usb_yolov5x_web.py  # 组合入口
  README.md               # 操作与验证说明
```

## 分层职责

### 通用 Triton HTTP 客户端

`triton_http.py` 负责：

- 根据模型名、输入和请求输出构造 Triton binary tensor inference 请求。
- 使用 `urllib.request` 发送 HTTP 请求。
- 解析 Triton binary tensor 响应中的命名原始输出缓冲区。

它不应该了解 YOLO、NV12、摄像头、BPU 或网页渲染逻辑。

### 帧源层

`frame_source.py` 提供输入适配器。第一个适配器是通过 OpenCV 读取 USB 摄像头：

- 默认设备：`/dev/video0`
- 输出帧格式：BGR `numpy.ndarray`

该接口后续也可以扩展为视频文件、RTSP 或图片目录，而不影响 Triton 客户端和模型适配器代码。

### YOLOv5x 模型适配层

`yolov5x_adapter.py` 负责所有与 YOLOv5x 相关的知识：

- Triton 模型名：`yolov5x_bpu`
- 输入契约：
  - `data_y`: shape `[1, 672, 672, 1]`，datatype `UINT8`
  - `data_uv`: shape `[1, 336, 336, 2]`，datatype `UINT8`
- 输出契约：
  - `output`: shape `[1, 84, 84, 255]`，datatype `INT32`
  - `1310`: shape `[1, 42, 42, 255]`，datatype `INT32`
  - `1312`: shape `[1, 21, 21, 255]`，datatype `INT32`
- BGR 到 NV12 split-plane 的预处理
- 使用 `hrt_model_exec model_info` 读取输出 scale 元数据，完成 Python 侧 YOLO 解码
- 画框与标注

如果后处理需要 Triton 当前 backend 响应没有暴露的元数据，适配层可以通过板端元数据 API 加载同一份 `.hbm` 模型，但这条路径只用于后处理支持，不用于推理；推理仍必须走 Triton。

### 网页展示层

`web_stream.py` 负责：

- `/` HTML 页面
- `/stream.mjpg` MJPEG 流
- `/health` JSON 状态

它应只消费帧生成器，对 YOLO 细节保持无感知，只接收已经标注好的 JPEG 帧或状态消息。

### USB YOLO 网页入口

`run_usb_yolov5x_web.py` 负责把各层连接起来：

```text
UsbCameraSource -> Yolov5xAdapter -> TritonHttpClient -> MjpegServer
```

该入口只应包含参数解析和生命周期管理。

## 运行约定

先单独启动 Triton：

```bash
scripts/triton_e2e/start_bpu_triton.sh
```

再启动验证工具链：

```bash
python3 examples/realtime_yolo/run_usb_yolov5x_web.py \
  --video-device /dev/video0 \
  --triton-url http://127.0.0.1:8000 \
  --host 0.0.0.0 \
  --port 8080
```

浏览器访问：

```text
http://192.168.1.154:8080/
```

## 错误处理

- 摄像头无法打开时，应带着设备路径快速失败。
- Triton 不可达或返回非法响应时，网页服务应继续存活，并通过 `/health` 和画面叠加暴露错误。
- 后处理失败时，仍应继续输出摄像头画面，并叠加错误信息，便于继续排查后端传输链路。
- 收到 Ctrl-C 时，应干净释放摄像头资源。

## 测试与验证

实现应通过以下方式验证：

1. 通用 Triton 请求/响应处理单元测试。
2. 帧源预处理和 JPEG/网页工具单元测试。
3. 现有后端构建与测试：

   ```bash
   cmake -S bpu_backend -B build/bpu_backend -DBPU_BACKEND_BUILD_TRITON=ON
   cmake --build build/bpu_backend -j2
   ctest --test-dir build/bpu_backend --output-on-failure
   ```

4. 现有 Triton HTTP 端到端脚本：

   ```bash
   scripts/triton_e2e/start_bpu_triton.sh
   python3 scripts/triton_e2e/infer_yolov5x_http.py
   scripts/triton_e2e/stop_bpu_triton.sh
   ```

5. 通过 Triton 完成 USB 摄像头单帧验证：

   - 捕获 `/dev/video0` 帧
   - 使用 `Yolov5xAdapter` 预处理
   - 通过 `TritonHttpClient` 推理
   - 验证原始输出尺寸

6. 网页端点 smoke test：

   - 启动工具链
   - 获取 `/health`
   - 获取 `/stream.mjpg` 的首个 multipart chunk
   - 在浏览器中手动查看实时标注画面

## 第一阶段不包含的内容

- 把 YOLO 后处理放进 Triton backend 内部。
- 引入 WebSocket 或 Canvas 前端架构。
- 支持多路摄像头。
- 支持不带模型适配器的任意模型。
- 生产级认证、TLS 或对外发布能力。
- 容器化完整工具链。
