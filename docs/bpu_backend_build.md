# BPU 后端构建与验证

## 构建

```bash
cmake -S bpu_backend -B build/bpu_backend -DBPU_BACKEND_BUILD_TRITON=ON
cmake --build build/bpu_backend -j2
ctest --test-dir build/bpu_backend --output-on-failure
```

预期生成文件：

```text
build/bpu_backend/bpu_runner
build/bpu_backend/libtriton_bpu.so
```

## 元数据验证

```bash
./build/bpu_backend/bpu_runner \
  --model /opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm \
  --print-metadata
```

预期输入：

```text
data_y UINT8 [1,672,672,1]
data_uv UINT8 [1,336,336,2]
```

预期输出：

```text
output INT32 [1,84,84,255]
1310 INT32 [1,42,42,255]
1312 INT32 [1,21,21,255]
```

## 本地 runner 推理验证

```bash
python3 examples/yolov5x/prepare_input.py /path/to/image.jpg --out-dir tmp_inputs
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

## Triton 后端安装路径

当 `tritonserver` 可用时，后端应安装到：

```text
/opt/tritonserver/backends/bpu/libtriton_bpu.so
```

模型配置中使用：

```protobuf
backend: "bpu"
```
