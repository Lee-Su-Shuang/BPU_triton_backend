#include "bpu_backend/bpu_runtime.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <string>

#include <hobot/dnn/hb_dnn_status.h>

namespace bpu_backend {
namespace {

const TensorBuffer* FindInput(const std::vector<TensorBuffer>& inputs,
                              const std::string& name) {
  for (const auto& input : inputs) {
    if (input.meta.name == name) {
      return &input;
    }
  }
  return nullptr;
}

bool HasUsableStrides(const TensorMeta& meta);

void FillTensorProperties(const TensorMeta& meta, hbDNNTensor* tensor) {
  tensor->properties.validShape.numDimensions =
      static_cast<int32_t>(meta.shape.size());
  for (size_t i = 0; i < meta.shape.size(); ++i) {
    tensor->properties.validShape.dimensionSize[i] =
        static_cast<int32_t>(meta.shape[i]);
  }

  int64_t dense_stride = static_cast<int64_t>(ElementSizeBytes(meta.dtype));
  std::vector<int64_t> dense_strides(meta.shape.size(), 0);
  for (size_t i = meta.shape.size(); i > 0; --i) {
    dense_strides[i - 1] = dense_stride;
    dense_stride *= meta.shape[i - 1];
  }
  if (HasUsableStrides(meta)) {
    for (size_t i = 0; i < meta.strides.size(); ++i) {
      tensor->properties.stride[i] =
          meta.strides[i] > 0 ? meta.strides[i] : dense_strides[i];
    }
  } else {
    for (size_t i = 0; i < dense_strides.size(); ++i) {
      tensor->properties.stride[i] = dense_strides[i];
    }
  }

  switch (meta.dtype) {
    case DType::kUInt8:
      tensor->properties.tensorType = HB_DNN_TENSOR_TYPE_U8;
      break;
    case DType::kInt8:
      tensor->properties.tensorType = HB_DNN_TENSOR_TYPE_S8;
      break;
    case DType::kInt32:
      tensor->properties.tensorType = HB_DNN_TENSOR_TYPE_S32;
      break;
    case DType::kFloat32:
      tensor->properties.tensorType = HB_DNN_TENSOR_TYPE_F32;
      break;
    case DType::kUnsupported:
      tensor->properties.tensorType = HB_DNN_TENSOR_TYPE_MAX;
      break;
  }
  tensor->properties.alignedByteSize = meta.aligned_byte_size;
}

bool StrideByteSpan(const TensorMeta& meta, size_t* span) {
  if (meta.shape.empty() || meta.strides.size() != meta.shape.size()) {
    return false;
  }
  int64_t max_offset = 0;
  for (size_t i = 0; i < meta.shape.size(); ++i) {
    if (meta.shape[i] <= 0) {
      return false;
    }
    if (meta.shape[i] == 1) {
      if (meta.strides[i] < 0) {
        return false;
      }
      continue;
    }
    if (meta.strides[i] <= 0) {
      return false;
    }
    const int64_t extent = meta.shape[i] - 1;
    if (meta.strides[i] > std::numeric_limits<int64_t>::max() / extent) {
      return false;
    }
    const int64_t offset = extent * meta.strides[i];
    if (max_offset > std::numeric_limits<int64_t>::max() - offset) {
      return false;
    }
    max_offset += offset;
  }
  const size_t element_size = ElementSizeBytes(meta.dtype);
  const size_t max_offset_size = static_cast<size_t>(max_offset);
  if (element_size == 0 ||
      max_offset_size > std::numeric_limits<size_t>::max() - element_size) {
    return false;
  }
  *span = max_offset_size + element_size;
  return true;
}

bool HasUsableStrides(const TensorMeta& meta) {
  if (meta.strides.empty()) {
    return false;
  }
  size_t span = 0;
  return StrideByteSpan(meta, &span);
}

void CopyCompactToTensor(const TensorMeta& meta, const uint8_t* source,
                         uint8_t* destination) {
  if (!HasUsableStrides(meta)) {
    std::memcpy(destination, source, TensorByteSize(meta.dtype, meta.shape));
    return;
  }
  const size_t element_size = ElementSizeBytes(meta.dtype);
  size_t compact_offset = 0;
  std::function<void(size_t, size_t)> copy_dim = [&](size_t dim, size_t byte_offset) {
    if (dim == meta.shape.size()) {
      std::memcpy(destination + byte_offset, source + compact_offset, element_size);
      compact_offset += element_size;
      return;
    }
    for (int64_t i = 0; i < meta.shape[dim]; ++i) {
      copy_dim(dim + 1, byte_offset + static_cast<size_t>(i * meta.strides[dim]));
    }
  };
  copy_dim(0, 0);
}

void CopyTensorToCompact(const TensorMeta& meta, const uint8_t* source,
                         uint8_t* destination) {
  if (!HasUsableStrides(meta)) {
    std::memcpy(destination, source, TensorByteSize(meta.dtype, meta.shape));
    return;
  }
  const size_t element_size = ElementSizeBytes(meta.dtype);
  size_t compact_offset = 0;
  std::function<void(size_t, size_t)> copy_dim = [&](size_t dim, size_t byte_offset) {
    if (dim == meta.shape.size()) {
      std::memcpy(destination + compact_offset, source + byte_offset, element_size);
      compact_offset += element_size;
      return;
    }
    for (int64_t i = 0; i < meta.shape[dim]; ++i) {
      copy_dim(dim + 1, byte_offset + static_cast<size_t>(i * meta.strides[dim]));
    }
  };
  copy_dim(0, 0);
}

}  // namespace

UcpTensorBuffer::UcpTensorBuffer(UcpTensorBuffer&& other) noexcept {
  tensor_ = other.tensor_;
  allocated_size_ = other.allocated_size_;
  allocated_ = other.allocated_;
  other.tensor_ = hbDNNTensor{};
  other.allocated_size_ = 0;
  other.allocated_ = false;
}

UcpTensorBuffer& UcpTensorBuffer::operator=(
    UcpTensorBuffer&& other) noexcept {
  if (this != &other) {
    Release();
    tensor_ = other.tensor_;
    allocated_size_ = other.allocated_size_;
    allocated_ = other.allocated_;
    other.tensor_ = hbDNNTensor{};
    other.allocated_size_ = 0;
    other.allocated_ = false;
  }
  return *this;
}

UcpTensorBuffer::~UcpTensorBuffer() { Release(); }

Status UcpTensorBuffer::Allocate(const TensorMeta& meta) {
  Release();
  FillTensorProperties(meta, &tensor_);
  const size_t logical_size = TensorByteSize(meta.dtype, meta.shape);
  const size_t aligned_size = meta.aligned_byte_size > 0
                                  ? static_cast<size_t>(meta.aligned_byte_size)
                                  : 0;
  size_t stride_span = 0;
  const bool has_strides = !meta.strides.empty();
  const bool valid_stride_span = StrideByteSpan(meta, &stride_span);
  if (has_strides && !valid_stride_span) {
    return Status::InvalidArgument("invalid tensor strides for '" + meta.name + "'");
  }
  const size_t alloc_size = std::max({logical_size, aligned_size, stride_span});
  if (alloc_size == 0) {
    return Status::Unsupported("cannot allocate tensor '" + meta.name +
                               "' with zero bytes");
  }
  const int32_t rc = hbUCPMallocCached(&tensor_.sysMem, alloc_size, 0);
  if (rc != HB_DNN_SUCCESS) {
    return Status::RuntimeError("hbUCPMallocCached failed for tensor '" +
                                meta.name + "' with code " +
                                std::to_string(rc));
  }
  allocated_ = true;
  allocated_size_ = alloc_size;
  return Status::Ok();
}

void UcpTensorBuffer::Release() {
  if (allocated_) {
    hbUCPFree(&tensor_.sysMem);
    tensor_ = hbDNNTensor{};
    allocated_size_ = 0;
    allocated_ = false;
  }
}

Result<std::vector<TensorBuffer>> BpuRuntime::Infer(
    const std::vector<TensorBuffer>& inputs) const {
  std::vector<UcpTensorBuffer> input_tensors;
  input_tensors.reserve(model_.Inputs().size());

  for (const auto& expected : model_.Inputs()) {
    const TensorBuffer* actual = FindInput(inputs, expected.name);
    if (actual == nullptr) {
      return Status::InvalidArgument("missing input tensor: " + expected.name);
    }
    if (actual->meta.dtype != expected.dtype || actual->meta.shape != expected.shape) {
      return Status::InvalidArgument("input tensor metadata mismatch: " +
                                     expected.name);
    }
    Status size_status = ValidateTensorByteSize(expected, actual->bytes.size());
    if (!size_status.ok()) {
      return size_status;
    }

    UcpTensorBuffer tensor;
    Status alloc_status = tensor.Allocate(expected);
    if (!alloc_status.ok()) {
      return alloc_status;
    }
    CopyCompactToTensor(expected, actual->bytes.data(),
                        static_cast<uint8_t*>(tensor.data()));
    hbUCPMemFlush(&tensor.tensor()->sysMem, HB_SYS_MEM_CACHE_CLEAN);
    input_tensors.push_back(std::move(tensor));
  }

  std::vector<UcpTensorBuffer> output_tensors;
  output_tensors.reserve(model_.Outputs().size());
  for (const auto& output_meta : model_.Outputs()) {
    UcpTensorBuffer tensor;
    Status alloc_status = tensor.Allocate(output_meta);
    if (!alloc_status.ok()) {
      return alloc_status;
    }
    output_tensors.push_back(std::move(tensor));
  }

  std::vector<hbDNNTensor> hb_inputs;
  hb_inputs.reserve(input_tensors.size());
  for (const auto& tensor : input_tensors) {
    hb_inputs.push_back(*tensor.tensor());
  }
  std::vector<hbDNNTensor> hb_outputs;
  hb_outputs.reserve(output_tensors.size());
  for (const auto& tensor : output_tensors) {
    hb_outputs.push_back(*tensor.tensor());
  }

  const int32_t rc =
      hbDNNInferV2(nullptr, hb_outputs.data(), hb_inputs.data(), model_.Handle());
  if (rc != HB_DNN_SUCCESS) {
    return Status::RuntimeError(HbDnnErrorMessage("hbDNNInferV2", rc));
  }

  std::vector<TensorBuffer> outputs;
  outputs.reserve(model_.Outputs().size());
  for (size_t i = 0; i < model_.Outputs().size(); ++i) {
    const TensorMeta& meta = model_.Outputs()[i];
    hbUCPMemFlush(&output_tensors[i].tensor()->sysMem,
                  HB_SYS_MEM_CACHE_INVALIDATE);
    TensorBuffer buffer;
    buffer.meta = meta;
    const size_t logical_size = TensorByteSize(meta.dtype, meta.shape);
    buffer.bytes.resize(logical_size);
    CopyTensorToCompact(meta, static_cast<const uint8_t*>(output_tensors[i].data()),
                        buffer.bytes.data());
    outputs.push_back(std::move(buffer));
  }

  return outputs;
}

}  // namespace bpu_backend
