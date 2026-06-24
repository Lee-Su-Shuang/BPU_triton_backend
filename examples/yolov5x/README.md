# YOLOv5x BPU Triton Backend 示例

本示例使用板端提供的模型：

```text
/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

## 准备模型仓库

```bash
cd /home/sunrise/BPU_triton_backend
./examples/yolov5x/make_symlink.sh
```

该脚本会在 Triton model repository 中创建指向板端 `.hbm` 模型的符号链接。

## 从图片准备 NV12 split-plane 输入

```bash
python3 examples/yolov5x/prepare_input.py /path/to/image.jpg --out-dir tmp_inputs
```

生成的输入文件包括：

```text
tmp_inputs/data_y.bin
tmp_inputs/data_uv.bin
```

## 不经过 Triton 的本地 BPU 推理

```bash
./build/bpu_backend/bpu_runner \
  --model /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm \
  --input data_y=tmp_inputs/data_y.bin \
  --input data_uv=tmp_inputs/data_uv.bin \
  --dump-output tmp_outputs
```

预期输出文件：

```text
tmp_outputs/output.bin
tmp_outputs/1310.bin
tmp_outputs/1312.bin
```

## 使用 Triton Server

当 `tritonserver` 可用时，可以使用以下方式启动：

```bash
tritonserver \
  --backend-directory=/path/to/install/backends \
  --model-repository=/home/sunrise/BPU_triton_backend/examples/yolov5x/model_repository
```

在本仓库中也可以直接使用封装脚本：

```bash
scripts/triton_e2e/start_bpu_triton.sh
python3 scripts/triton_e2e/infer_yolov5x_http.py
scripts/triton_e2e/stop_bpu_triton.sh
```
