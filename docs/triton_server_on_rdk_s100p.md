# RDK S100P 上的 Triton Server

`tritonserver` 默认不在板端 `PATH` 中：

```bash
command -v tritonserver
```

初始结果通常为：

```text
tritonserver not found in PATH
```

## 已验证的独立包

本仓库在板端验证过官方 Triton Server aarch64 独立包：

```text
tritonserver-2.69.0+nv26.05-51748363-cu132-cp312-manylinux_2_28-aarch64.zip
```

下载位置：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-2.69.0-aarch64.zip
```

解压后目录：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver
```

重要说明：该官方包是 NVIDIA `+nv26.05-cu132` 构建，动态依赖 `libcudart.so.13` 和 `libdcgm.so.4`。RDK S100P 不是 NVIDIA CUDA 平台，因此仓库提供了本地无 GPU stub 工作流用于验证。这里的 stub 仅通过 `LD_LIBRARY_PATH` 供该 Triton Server 进程使用，不做全局安装。

## 构建本地无 GPU stub

在仓库根目录执行：

```bash
scripts/triton_e2e/build_no_gpu_stubs.sh /home/sunrise/triton_backend_BPU/tritonserver_dist
```

会生成：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/stubs/libcudart.so.13
/home/sunrise/triton_backend_BPU/tritonserver_dist/stubs/libdcgm.so.4
```

使用这些 stub 启动 Triton 时，可能看到以下警告：

```text
Unable to allocate pinned system memory, pinned memory pool will not be available: no CUDA-capable device is detected
CUDA memory pool disabled
CudaDriverHelper has not been initialized.
```

这些警告在 BPU 后端验证中是可接受的，因为 BPU 后端使用 CPU 请求缓冲区和 Horizon/D-Robotics DNN/HBRT runtime，而不是 CUDA。

## 启动 Triton 与 BPU 后端

在仓库根目录执行：

```bash
scripts/triton_e2e/start_bpu_triton.sh
```

脚本会：

1. 构建 `build/bpu_backend/libtriton_bpu.so`。
2. 安装到：

   ```text
   /home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver/backends/bpu/libtriton_bpu.so
   ```

3. 执行 `examples/yolov5x/make_symlink.sh`。
4. 使用以下参数启动 Triton Server：

   ```text
   --backend-directory=/home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver/backends
   --model-repository=<repo>/examples/yolov5x/model_repository
   --http-port=8000
   --grpc-port=8001
   --metrics-port=8002
   ```

运行日志：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-bpu.log
```

PID 文件：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-bpu.pid
```

## 验证 HTTP 推理

在仓库根目录执行：

```bash
scripts/triton_e2e/infer_yolov5x_http.py
```

预期输出尺寸：

```text
raw_bytes 9446220 parsed_bytes 9446220 sizes {'output': 7197120, '1310': 1799280, '1312': 449820}
```

本仓库在 2026-06-23 的验证中成功加载了模型：

```text
| Model       | Version | Status |
| yolov5x_bpu | 1       | READY  |
```

元数据接口返回：

```text
inputs: data_y UINT8 [1,672,672,1], data_uv UINT8 [1,336,336,2]
outputs: output INT32 [1,84,84,255], 1310 INT32 [1,42,42,255], 1312 INT32 [1,21,21,255]
```

这说明 HTTP 推理请求已经通过 Triton 和 `bpu` backend 成功完成，输出字节大小与本地 `bpu_runner` 验证一致。

## 停止 Triton

```bash
scripts/triton_e2e/stop_bpu_triton.sh
```

## 生产部署注意事项

当前验证路径是：RDK S100P 上使用官方 aarch64 NVIDIA 相关独立包，再配合本地无 GPU stub 完成 BPU 后端的端到端验证。若要做生产部署，建议使用以下任一种方案：

1. 适用于 aarch64 Ubuntu 22.04 的纯 CPU / 非 CUDA Triton Server 构建。
2. 不依赖 NVIDIA GPU / DCGM 的 Triton Server 构建。
3. 仅包含 BPU 后端运行时依赖的维护型部署镜像。
