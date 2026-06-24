#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bpu_backend/bpu_status.h"

namespace bpu_backend {

enum class DType {
  kUInt8,
  kInt8,
  kInt32,
  kFloat32,
  kUnsupported,
};

using TensorShape = std::vector<int64_t>;

struct TensorMeta {
  std::string name;
  DType dtype = DType::kUnsupported;
  TensorShape shape;
  int64_t aligned_byte_size = -1;
  std::vector<int64_t> strides;
};

struct TensorBuffer {
  TensorMeta meta;
  std::vector<uint8_t> bytes;
};

size_t ElementSizeBytes(DType dtype);
int64_t ShapeElementCount(const TensorShape& shape);
size_t TensorByteSize(DType dtype, const TensorShape& shape);
std::string DTypeName(DType dtype);
std::string ShapeToString(const TensorShape& shape);
Status ValidateTensorByteSize(const TensorMeta& meta, size_t actual_bytes);
std::string MetadataTable(const std::vector<TensorMeta>& inputs,
                          const std::vector<TensorMeta>& outputs);
Result<std::string> TritonConfigTemplate(const std::string& model_name,
                                         const std::vector<TensorMeta>& inputs,
                                         const std::vector<TensorMeta>& outputs);

}  // namespace bpu_backend
