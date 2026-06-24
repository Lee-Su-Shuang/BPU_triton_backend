# 实时 YOLOv5x Triton BPU 后端验证工具链

本示例使用实时 USB 摄像头输入和浏览器可视化来验证 Triton `bpu` backend。推理路径始终经过 Triton HTTP、Triton Server、`libtriton_bpu.so` 和 S100 BPU runtime。

## 架构

- `triton_http.py`：通用 Triton HTTP 二进制 tensor 客户端。
- `frame_source.py`：帧源模块，当前默认使用 `/dev/video0` USB 摄像头。
- `yolov5x_adapter.py`：YOLOv5x 专属输入输出契约、预处理、后处理和标注。
- `web_stream.py`：通用 MJPEG 视频流和健康检查接口。
- `run_usb_yolov5x_web.py`：组合入口。

## 启动 Triton

在仓库根目录执行：

```bash
scripts/triton_e2e/start_bpu_triton.sh
```

## 启动验证工具链

```bash
python3 examples/realtime_yolo/run_usb_yolov5x_web.py \
  --video-device /dev/video0 \
  --triton-url http://127.0.0.1:8000 \
  --host 0.0.0.0 \
  --port 8080
```

浏览器打开：

```text
http://192.168.1.154:8080/
```

健康检查接口：

```text
http://192.168.1.154:8080/health
```

## 停止

在验证工具链终端按 Ctrl-C，然后停止 Triton：

```bash
scripts/triton_e2e/stop_bpu_triton.sh
```

## 可移植性说明

Triton 客户端和网页层是模型无关的。YOLOv5x 预处理、后处理和 S100 板端路径都隔离在 `yolov5x_adapter.py` 中。若要验证其他模型，可以新增对应的 model adapter，定义自己的 Triton 输入输出契约和后处理逻辑。

`hrt_model_exec model_info` 用于读取 YOLOv5x adapter 所需的输出 scale 元数据，但推理仍然通过 Triton HTTP 和 `bpu` backend 完成。
