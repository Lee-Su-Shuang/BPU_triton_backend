# CLAUDE.md

本文件是 `/home/sunrise/BPU_triton_backend` 项目的集中总结与后续维护说明。用户准备删除当前目录时，应优先保留或参考本文件；项目源码已推送到 GitHub：

```text
git@github.com:Lee-Su-Shuang/BPU_triton_backend.git
branch: main
最近已验证文档提交: 73350603ece0aac2325c2cb4c73992d0df58c362
```

## 项目概述

本项目实现了一个面向 D-Robotics / Horizon Robotics RDK S100P 板载 BPU 的 Triton Server 自定义后端。后端名称为 `bpu`，编译产物为：

```text
libtriton_bpu.so
```

目标是让 Triton Server 能够通过自定义 `bpu` backend 加载并运行 Horizon/D-Robotics `.hbm` 模型，使用板端 DNN/HBRT runtime 调用 S100P BPU 执行推理，并通过 Triton 标准 HTTP inference API 对外提供服务。

项目同时包含 YOLOv5x `.hbm` 模型的验证配置、Triton HTTP 端到端验证脚本、USB 摄像头实时网页验证工具链和一键验证脚本。

## 运行环境与外部依赖

项目已在以下环境验证：

```text
Board:         D-Robotics RDK S100 V1P0
OS:            Ubuntu 22.04.5 LTS / RDK OS 4.0.2-Beta
Triton Server: 2.69.0
DNN:           3.13.6
HBRT:          4.7.5
BPULib:        2.1.2
```

运行时依赖：

```text
YOLOv5x model:       /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
Triton Server dist:  /home/sunrise/triton_backend_BPU/tritonserver_dist
BPU runtime libs:    /usr/hobot/lib/libdnn.so, /usr/hobot/lib/libhbrt4.so
BPU devices:         /dev/bpu, /dev/bpu_core0
USB camera:          /dev/video0
```

仓库不直接提交板载 `.hbm` 模型，也不提交 Triton Server 二进制分发包。YOLOv5x model repository 通过符号链接引用板端模型：

```text
examples/yolov5x/model_repository/yolov5x_bpu/1/model.hbm -> /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

## 目录结构

```text
BPU_triton_backend/
  README.md                  项目入口说明
  MANIFEST.md                文件清单、外部依赖和版本信息
  CLAUDE.md                  本集中总结文件
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

## 核心架构

整体推理链路：

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

实时网页验证链路：

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

## C++ 后端实现摘要

主要源码位于：

```text
bpu_backend/
```

关键文件：

```text
bpu_backend/include/bpu_backend/bpu_status.h        Result/Status 错误处理
bpu_backend/include/bpu_backend/bpu_tensor.h        tensor metadata 和 raw buffer 描述
bpu_backend/include/bpu_backend/bpu_model.h         .hbm 模型加载和 metadata 读取
bpu_backend/include/bpu_backend/bpu_runtime.h       BPU 推理封装
bpu_backend/include/bpu_backend/bpu_runner_args.h   CLI runner 参数解析

bpu_backend/src/bpu_tensor.cc
bpu_backend/src/bpu_model.cc
bpu_backend/src/bpu_runtime.cc
bpu_backend/src/bpu_runner_args.cc
bpu_backend/src/triton_backend.cc                   Triton backend API 实现

bpu_backend/tools/bpu_runner.cc                     本地调试工具
bpu_backend/tests/                                  C++ 单元测试
```

`triton_backend.cc` 实现的 Triton backend API：

```text
TRITONBACKEND_ModelInitialize
TRITONBACKEND_ModelFinalize
TRITONBACKEND_ModelInstanceInitialize
TRITONBACKEND_ModelInstanceFinalize
TRITONBACKEND_ModelInstanceExecute
```

执行流程：

1. Triton 加载模型时，backend 从 model repository 找到 `.hbm`。
2. `BpuModel` 加载 `.hbm` 并读取输入 / 输出 metadata。
3. 请求进入 `ModelInstanceExecute`。
4. backend 收集 Triton 请求输入缓冲区。
5. 调用 `BpuRuntime::Infer`。
6. 将输出原始字节写入 Triton response output buffers。
7. Triton 通过 HTTP / gRPC 返回结果。

后端中的关键保护：

- 仅接受 CPU input buffer。
- 检查 input buffer size，避免 overflow。
- 检查 output buffer memory type。
- `ModelSetState` 失败时释放 state，避免泄漏。
- response output 创建失败时正确返回 Triton error。

## Python 实时验证工具链

位置：

```text
examples/realtime_yolo/
```

模块职责：

```text
triton_http.py          通用 Triton HTTP binary tensor 客户端
frame_source.py         USB 摄像头等帧源；默认 /dev/video0
yolov5x_adapter.py      YOLOv5x 输入输出契约、预处理、后处理和标注
web_stream.py           MJPEG 网页流和 /health 接口
run_usb_yolov5x_web.py  组合入口
```

设计原则：

- 推理必须经过 Triton HTTP 和 `bpu` backend，不能绕过 Triton。
- C++ backend 保持通用，只返回 `.hbm` raw output tensors。
- YOLOv5x 预处理、后处理、scale metadata、可视化均放在 Python adapter 层。
- `hrt_model_exec model_info` 只用于读取后处理所需 scale metadata，不用于替代 Triton 推理。

## YOLOv5x 模型配置

位置：

```text
examples/yolov5x/
```

Triton 模型配置：

```text
model name: yolov5x_bpu
backend:    bpu
inputs:     data_y, data_uv
outputs:    output, 1310, 1312
```

输入：

```text
data_y  UINT8 [1,672,672,1]
data_uv UINT8 [1,336,336,2]
```

输出：

```text
output INT32 [1,84,84,255]
1310   INT32 [1,42,42,255]
1312   INT32 [1,21,21,255]
```

## 常用命令

### 克隆 / 恢复项目

```bash
git clone git@github.com:Lee-Su-Shuang/BPU_triton_backend.git /home/sunrise/BPU_triton_backend
cd /home/sunrise/BPU_triton_backend
```

如果模型符号链接不存在：

```bash
./examples/yolov5x/make_symlink.sh
```

### 一键完整验证

```bash
cd /home/sunrise/BPU_triton_backend
./verify_all.sh
```

成功标志：

```text
All BPU Triton backend deliverable checks passed.
```

### 只构建和测试 C++ 后端

```bash
cmake -S bpu_backend -B build/bpu_backend -DBPU_BACKEND_BUILD_TRITON=ON
cmake --build build/bpu_backend -j2
ctest --test-dir build/bpu_backend --output-on-failure
```

预期生成：

```text
build/bpu_backend/libtriton_bpu.so
build/bpu_backend/bpu_runner
```

### 打印 `.hbm` 模型 metadata

```bash
./build/bpu_backend/bpu_runner \
  --model /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm \
  --print-metadata
```

### 准备 YOLOv5x 输入并本地 BPU 推理

```bash
python3 examples/yolov5x/prepare_input.py /path/to/image.jpg --out-dir tmp_inputs
./build/bpu_backend/bpu_runner \
  --model /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm \
  --input data_y=tmp_inputs/data_y.bin \
  --input data_uv=tmp_inputs/data_uv.bin \
  --dump-output tmp_outputs
```

预期输出：

```text
tmp_outputs/output.bin
tmp_outputs/1310.bin
tmp_outputs/1312.bin
```

### 启动 Triton HTTP 端到端验证

```bash
scripts/triton_e2e/start_bpu_triton.sh
python3 scripts/triton_e2e/infer_yolov5x_http.py
scripts/triton_e2e/stop_bpu_triton.sh
```

预期输出尺寸：

```text
raw_bytes 9446220 parsed_bytes 9446220 sizes {'output': 7197120, '1310': 1799280, '1312': 449820}
```

### 启动实时网页验证

```bash
./run_realtime_yolo.sh
```

浏览器访问：

```text
http://192.168.1.154:8080/
```

停止：

```bash
./stop_services.sh
```

### 查看服务状态

```bash
curl http://127.0.0.1:8000/v2/health/ready
curl http://127.0.0.1:8000/v2/models/yolov5x_bpu
curl http://127.0.0.1:8080/health
```

## Triton Server 说明

`tritonserver` 默认不在板端 `PATH` 中。本项目验证使用的 Triton Server 分发包位于：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist
```

使用的是官方 aarch64 standalone 包，但该包带 NVIDIA CUDA/DCGM 动态依赖。RDK S100P 不是 CUDA 平台，因此项目提供本地无 GPU stub：

```bash
scripts/triton_e2e/build_no_gpu_stubs.sh /home/sunrise/triton_backend_BPU/tritonserver_dist
```

生成：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/stubs/libcudart.so.13
/home/sunrise/triton_backend_BPU/tritonserver_dist/stubs/libdcgm.so.4
```

这些 stub 仅通过 `LD_LIBRARY_PATH` 供验证用 Triton 进程加载，不全局安装。

启动脚本会把 `libtriton_bpu.so` 安装到：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/server/tritonserver/backends/bpu/libtriton_bpu.so
```

Triton 运行日志：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-bpu.log
```

PID 文件：

```text
/home/sunrise/triton_backend_BPU/tritonserver_dist/tritonserver-bpu.pid
```

## 已验证结果

最近完整验证命令：

```bash
./verify_all.sh
```

验证结果：

```text
C++ tests:       7/7 passed
Python tests:    17 passed
Triton HTTP e2e: output sizes 7197120 / 1799280 / 449820
Camera smoke:    /dev/video0 opened and read frame (480, 640, 3)
Web smoke:       /stream.mjpg returned multipart JPEG data
```

`verify_all.sh` 成功时输出：

```text
All BPU Triton backend deliverable checks passed.
```

## 当前 Git 状态记录

项目已推送到：

```text
git@github.com:Lee-Su-Shuang/BPU_triton_backend.git
```

关键提交：

```text
240db1c11496c8d5753cc7619485a39f85405f6b  feat: add BPU Triton backend package
73350603ece0aac2325c2cb4c73992d0df58c362  docs: rewrite Markdown for public Chinese repository
```

如果本文件随后也被提交，应以最新远端 `main` 为准。

## 删除本地目录前注意事项

如果要删除 `/home/sunrise/BPU_triton_backend`：

1. 先确认最新内容已经推送到 GitHub。
2. 不需要保留 `build/`、`.pytest_cache/`、`__pycache__/` 等本地产物。
3. 仓库依赖的板端资源不会随 GitHub 仓库恢复，需要板端仍然存在：
   - `/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm`
   - `/home/sunrise/triton_backend_BPU/tritonserver_dist`
   - `/usr/hobot/lib/libdnn.so`
   - `/usr/hobot/lib/libhbrt4.so`
   - `/dev/bpu`, `/dev/bpu_core0`
   - `/dev/video0`
4. 恢复后先运行 `./examples/yolov5x/make_symlink.sh`，再运行 `./verify_all.sh`。

## 后续维护建议

- 后端保持通用：不要把 YOLOv5x 后处理写进 `libtriton_bpu.so`。
- 新模型应新增独立 adapter 和 Triton model repository 配置。
- 涉及 runtime / device / Triton Server 路径的改动，要同步更新 `README.md`、`MANIFEST.md` 和本文件。
- 对外公开文档保持中文正式仓库风格，避免出现课堂提交、临时演示或面向个人评审的措辞。
- 每次声称完成前，运行相应验证命令；涉及后端或脚本改动时优先运行 `./verify_all.sh`。
