# Demo Guide

本文档用于现场向老师演示本项目。

## 1. 进入交付目录

```bash
cd /home/sunrise/BPU_triton_backend
```

## 2. 说明项目目标

可以先向老师说明：

```text
本项目实现了一个 Triton Server 自定义 C++ backend，backend 名称为 bpu。
它可以通过 Horizon/D-Robotics DNN/HBRT runtime 调用 RDK S100P 板载 BPU，运行 .hbm 模型。
项目提供了 YOLOv5x 模型的 Triton HTTP e2e 验证，以及 USB 摄像头实时网页验证。
```

## 3. 完整验收演示

运行：

```bash
./verify_all.sh
```

老师可以看到脚本依次完成：

```text
[1/7] Build BPU backend
[2/7] Run C++ backend tests
[3/7] Run realtime harness and package Python tests
[4/7] Run Triton HTTP e2e
[5/7] Check USB camera capture
[6/7] Start realtime web harness for endpoint smoke
[7/7] Smoke web health and MJPEG stream
```

成功标志：

```text
All BPU Triton backend deliverable checks passed.
```

## 4. 实时网页演示

运行：

```bash
./run_realtime_yolo.sh
```

浏览器打开：

```text
http://192.168.1.154:8080/
```

页面显示 USB 摄像头实时画面和 YOLO 检测框。

## 5. 说明实时链路

可以向老师说明实时链路：

```text
USB camera /dev/video0
  -> Python realtime_yolo harness
  -> Triton HTTP request
  -> Triton Server
  -> bpu backend libtriton_bpu.so
  -> S100P BPU
  -> yolov5x_672x672_nv12.hbm
  -> raw output tensors
  -> YOLO postprocess
  -> MJPEG browser stream
```

其中推理一定经过 Triton Server 和自定义 `bpu` backend，网页部分只是验证和展示。

## 6. 查看 Triton 状态

另开一个终端：

```bash
curl http://127.0.0.1:8000/v2/health/ready
curl http://127.0.0.1:8000/v2/models/yolov5x_bpu
curl http://127.0.0.1:8080/health
```

`/v2/models/yolov5x_bpu` 会显示：

```text
platform: bpu
inputs:  data_y, data_uv
outputs: output, 1310, 1312
```

## 7. 停止演示

```bash
./stop_services.sh
```

## 8. 老师可能关注的问题

### 是否真的走了 BPU？

是。Triton 日志中会看到：

```text
Backend: bpu
libtriton_bpu.so
BPULib version 2.1.2
DNN 3.13.6 / HBRT 4.7.5
model yolov5x_bpu READY
```

### 模型在哪里？

模型不复制进交付目录，使用板载模型：

```text
/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

Triton model repository 里通过 symlink 引用它。

### 为什么网页不卡？

实时网页使用异步 pipeline：

```text
capture thread:   持续读取最新帧
inference thread: 持续用最新帧跑 Triton/BPU
web thread:       输出最新画面和最新检测结果
```

这样视频显示不会被单次 BPU/Triton 推理阻塞。
