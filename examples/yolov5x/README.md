# YOLOv5x BPU Triton Backend Example

This example uses the board-provided model:

```text
/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm
```

Prepare the model repository:

```bash
cd /home/sunrise/triton_backend_BPU/backend/.claude/worktrees/bpu-triton-backend
./examples/yolov5x/make_symlink.sh
```

Prepare raw NV12 split-plane inputs from an image:

```bash
python3 examples/yolov5x/prepare_input.py /path/to/image.jpg --out-dir tmp_inputs
```

Run local BPU inference without Triton:

```bash
./build/bpu_backend/bpu_runner \
  --model /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm \
  --input data_y=tmp_inputs/data_y.bin \
  --input data_uv=tmp_inputs/data_uv.bin \
  --dump-output tmp_outputs
```

Expected output files:

```text
tmp_outputs/output.bin
tmp_outputs/1310.bin
tmp_outputs/1312.bin
```

When `tritonserver` is available, run it with:

```bash
tritonserver \
  --backend-directory=/path/to/install/backends \
  --model-repository=/home/sunrise/triton_backend_BPU/backend/.claude/worktrees/bpu-triton-backend/examples/yolov5x/model_repository
```
