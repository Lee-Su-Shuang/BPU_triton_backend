# 运行验证说明

## 一键完整验证

在仓库根目录运行：

```bash
cd /home/sunrise/BPU_triton_backend
./verify_all.sh
```

这是本项目的最终验收入口。

## verify_all.sh 的执行内容

### 1. 外部依赖检查

检查以下资源是否可用：

```text
/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
/home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver/bin/tritonserver
/usr/hobot/lib/libdnn.so
/usr/hobot/lib/libhbrt4.so
/dev/bpu
/dev/video0
```

### 2. 编译 C++ backend

```bash
cmake -S bpu_backend -B build/bpu_backend -DBPU_BACKEND_BUILD_TRITON=ON
cmake --build build/bpu_backend -j2
```

预期生成：

```text
build/bpu_backend/libtriton_bpu.so
build/bpu_backend/bpu_runner
```

### 3. 运行 C++ 单元测试

```bash
ctest --test-dir build/bpu_backend --output-on-failure
```

测试覆盖：

- tensor metadata
- runner args
- config template
- `.hbm` 模型加载
- BPU runtime wrapper
- runner behavior
- config stdout

预期结果：

```text
100% tests passed, 0 tests failed out of 7
```

### 4. 运行 Python / package 测试

```bash
python3 -m pytest examples/realtime_yolo tests -v
```

测试覆盖：

- Triton HTTP 二进制 tensor 请求 / 响应
- USB frame validation
- YOLOv5x 输入输出契约
- YOLO 输出 scale 元数据解析
- MJPEG 页面与重连逻辑
- 异步 pipeline 不阻塞网页帧输出
- 脚本启动前清理旧服务

预期结果：

```text
17 passed
```

### 5. Triton HTTP 端到端验证

脚本：

```bash
scripts/triton_e2e/start_bpu_triton.sh
python3 scripts/triton_e2e/infer_yolov5x_http.py
```

验证内容：

- 编译并安装 `libtriton_bpu.so` 到 Triton backend 目录。
- 准备 `yolov5x_bpu` model repository。
- 启动 Triton Server。
- 通过 HTTP 调用 `/v2/models/yolov5x_bpu/infer`。
- 检查原始输出 tensor 的字节大小。

预期输出：

```text
raw_bytes 9446220 parsed_bytes 9446220 sizes {
  'output': 7197120,
  '1310': 1799280,
  '1312': 449820
}
```

### 6. USB 摄像头 smoke test

验证输出：

```text
camera_opened True
camera_read True shape (480, 640, 3)
```

### 7. Web endpoint smoke test

临时启动实时网页工具链，验证：

```text
/health       返回 JSON
/stream.mjpg  返回 multipart JPEG data
```

预期 stream 中包含：

```text
--frame
Content-Type: image/jpeg
JFIF
```

## 手动分步测试

### 只测 C++ backend

```bash
cmake -S bpu_backend -B build/bpu_backend -DBPU_BACKEND_BUILD_TRITON=ON
cmake --build build/bpu_backend -j2
ctest --test-dir build/bpu_backend --output-on-failure
```

### 只测 Triton HTTP

```bash
scripts/triton_e2e/start_bpu_triton.sh
python3 scripts/triton_e2e/infer_yolov5x_http.py
scripts/triton_e2e/stop_bpu_triton.sh
```

### 只测实时网页

```bash
./run_realtime_yolo.sh
```

浏览器打开：

```text
http://192.168.1.154:8080/
```

停止：

```bash
./stop_services.sh
```

## 验证结论

如果 `./verify_all.sh` 通过，说明：

```text
源码可编译
单元测试通过
Triton 能加载 bpu backend
Triton 能通过 HTTP 调用 BPU backend
BPU 能运行 YOLOv5x .hbm 模型
USB camera 实时网页验证能输出视频流
```
