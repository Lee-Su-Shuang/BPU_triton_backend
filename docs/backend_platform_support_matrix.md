# 平台支持矩阵说明

> 说明：本文件保留原始上游 Triton 平台兼容性信息，用于参考不同后端在不同平台上的支持情况。

## Ubuntu 22.04

下表描述了各后端在不同平台上的推理支持情况。

| 后端 | x86 | ARM-SBSA |
| --- | --- | --- |
| TensorRT | GPU / CPU | GPU / CPU |
| ONNX Runtime | GPU / CPU | GPU / CPU |
| TensorFlow | GPU / CPU | GPU / CPU |
| PyTorch | GPU / CPU | GPU / CPU |
| OpenVINO | CPU | 不支持 GPU / CPU |
| Python | GPU / CPU | GPU / CPU |
| DALI | GPU / CPU | GPU / CPU |
| FIL | GPU / CPU | 不支持 |
| TensorRT-LLM | GPU | GPU |
| vLLM | GPU / CPU | 不支持 |

## Jetson JetPack

以下后端目前支持 Jetson JetPack：

| 后端 | Jetson |
| --- | --- |
| TensorRT | GPU |
| ONNX Runtime | GPU / CPU |
| TensorFlow | GPU / CPU |
| PyTorch | GPU / CPU |
| Python | CPU |

更多信息可参考上游 Triton 文档。

## AWS Inferentia

当前，AWS Inferentia 的推理主要通过 Python backend 实现，由部署的 Python 脚本调用 AWS Neuron SDK 完成。

## 说明

1. Python Backend 支持的设备范围以 Triton 的描述为准。Python backend 中运行的脚本可以在具备相应 Python API 的硬件上执行推理。
2. ARM-SBSA 下的部分操作支持仍有局限。
