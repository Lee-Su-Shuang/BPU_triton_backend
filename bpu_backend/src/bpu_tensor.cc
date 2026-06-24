#include "bpu_backend/bpu_tensor.h"

#include <limits>
#include <sstream>

namespace bpu_backend {

size_t ElementSizeBytes(DType dtype) {
  switch (dtype) {
    case DType::kUInt8:
    case DType::kInt8:
      return 1;
    case DType::kInt32:
    case DType::kFloat32:
      return 4;
    case DType::kUnsupported:
      return 0;
  }
  return 0;
}

int64_t ShapeElementCount(const TensorShape& shape) {
  if (shape.empty()) {
    return 0;
  }
  int64_t count = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    if (count > std::numeric_limits<int64_t>::max() / dim) {
      return 0;
    }
    count *= dim;
  }
  return count;
}

size_t TensorByteSize(DType dtype, const TensorShape& shape) {
  const size_t element_size = ElementSizeBytes(dtype);
  const int64_t element_count = ShapeElementCount(shape);
  if (element_size == 0 || element_count <= 0) {
    return 0;
  }
  const auto element_count_size = static_cast<size_t>(element_count);
  if (element_count_size > std::numeric_limits<size_t>::max() / element_size) {
    return 0;
  }
  return element_count_size * element_size;
}

std::string DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kUInt8:
      return "UINT8";
    case DType::kInt8:
      return "INT8";
    case DType::kInt32:
      return "INT32";
    case DType::kFloat32:
      return "FP32";
    case DType::kUnsupported:
      return "UNSUPPORTED";
  }
  return "UNSUPPORTED";
}

std::string ShapeToString(const TensorShape& shape) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

Status ValidateTensorByteSize(const TensorMeta& meta, size_t actual_bytes) {
  const size_t expected = TensorByteSize(meta.dtype, meta.shape);
  if (expected == 0) {
    return Status::Unsupported("unsupported tensor dtype or shape for '" +
                               meta.name + "'");
  }
  if (actual_bytes != expected) {
    return Status::InvalidArgument(
        "tensor '" + meta.name + "' byte size mismatch: expected " +
        std::to_string(expected) + ", got " + std::to_string(actual_bytes));
  }
  return Status::Ok();
}

namespace {

Result<std::string> ProtoDType(DType dtype) {
  switch (dtype) {
    case DType::kUInt8:
      return std::string("TYPE_UINT8");
    case DType::kInt8:
      return std::string("TYPE_INT8");
    case DType::kInt32:
      return std::string("TYPE_INT32");
    case DType::kFloat32:
      return std::string("TYPE_FP32");
    case DType::kUnsupported:
      return Status::Unsupported("unsupported tensor dtype");
  }
  return Status::Unsupported("unsupported tensor dtype");
}

std::string ProtoDims(const TensorShape& shape) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

}  // namespace

std::string MetadataTable(const std::vector<TensorMeta>& inputs,
                          const std::vector<TensorMeta>& outputs) {
  std::ostringstream out;
  out << "inputs:\n";
  for (const auto& input : inputs) {
    out << "  " << input.name << " " << DTypeName(input.dtype) << " "
        << ShapeToString(input.shape) << " bytes="
        << TensorByteSize(input.dtype, input.shape) << " aligned="
        << input.aligned_byte_size << "\n";
  }
  out << "outputs:\n";
  for (const auto& output : outputs) {
    out << "  " << output.name << " " << DTypeName(output.dtype) << " "
        << ShapeToString(output.shape) << " bytes="
        << TensorByteSize(output.dtype, output.shape) << " aligned="
        << output.aligned_byte_size << "\n";
  }
  return out.str();
}

Result<std::string> TritonConfigTemplate(
    const std::string& model_name, const std::vector<TensorMeta>& inputs,
    const std::vector<TensorMeta>& outputs) {
  std::ostringstream out;
  out << "name: \"" << model_name << "\"\n";
  out << "backend: \"bpu\"\n";
  out << "max_batch_size: 0\n\n";
  out << "input [\n";
  for (const auto& input : inputs) {
    auto dtype = ProtoDType(input.dtype);
    if (!dtype.ok()) {
      return Status::Unsupported("unsupported tensor dtype for input '" +
                                 input.name + "'");
    }
    out << "  {\n"
        << "    name: \"" << input.name << "\"\n"
        << "    data_type: " << dtype.value() << "\n"
        << "    dims: " << ProtoDims(input.shape) << "\n"
        << "  }\n";
  }
  out << "]\n\n";
  out << "output [\n";
  for (const auto& output : outputs) {
    auto dtype = ProtoDType(output.dtype);
    if (!dtype.ok()) {
      return Status::Unsupported("unsupported tensor dtype for output '" +
                                 output.name + "'");
    }
    out << "  {\n"
        << "    name: \"" << output.name << "\"\n"
        << "    data_type: " << dtype.value() << "\n"
        << "    dims: " << ProtoDims(output.shape) << "\n"
        << "  }\n";
  }
  out << "]\n\n";
  out << "instance_group [\n"
      << "  {\n"
      << "    count: 1\n"
      << "    kind: KIND_CPU\n"
      << "  }\n"
      << "]\n";
  return out.str();
}

}  // namespace bpu_backend
