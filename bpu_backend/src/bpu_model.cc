#include "bpu_backend/bpu_model.h"

#include <filesystem>
#include <sstream>

#include <hobot/dnn/hb_dnn_status.h>

namespace bpu_backend {
namespace {

Result<std::vector<int64_t>> FromHbStrides(const hbDNNTensorProperties& properties) {
  if (properties.validShape.numDimensions < 0 ||
      properties.validShape.numDimensions > HB_DNN_TENSOR_MAX_DIMENSIONS) {
    return Status::InvalidArgument("invalid hbDNN tensor rank: " +
                                   std::to_string(properties.validShape.numDimensions));
  }
  std::vector<int64_t> strides;
  strides.reserve(static_cast<size_t>(properties.validShape.numDimensions));
  for (int32_t i = 0; i < properties.validShape.numDimensions; ++i) {
    strides.push_back(properties.stride[i]);
  }
  return strides;
}

Result<TensorMeta> BuildTensorMeta(const char* name,
                                   const hbDNNTensorProperties& properties) {
  TensorMeta meta;
  meta.name = name == nullptr ? "" : name;
  meta.dtype = FromHbDnnType(properties.tensorType);
  auto shape = FromHbShape(properties.validShape);
  if (!shape.ok()) {
    return shape.status();
  }
  meta.shape = shape.value();
  auto strides = FromHbStrides(properties);
  if (!strides.ok()) {
    return strides.status();
  }
  bool has_negative_stride = false;
  for (int64_t stride : strides.value()) {
    has_negative_stride = has_negative_stride || stride < 0;
  }
  if (!has_negative_stride) {
    meta.strides = strides.value();
  }
  meta.aligned_byte_size = properties.alignedByteSize;
  return meta;
}

}  // namespace

std::string HbDnnErrorMessage(const std::string& operation, int32_t code) {
  std::ostringstream out;
  out << operation << " failed with code " << code;
  const char* desc = hbDNNGetErrorDesc(code);
  if (desc != nullptr) {
    out << ": " << desc;
  }
  return out.str();
}

DType FromHbDnnType(int32_t hb_type) {
  switch (hb_type) {
    case HB_DNN_TENSOR_TYPE_U8:
      return DType::kUInt8;
    case HB_DNN_TENSOR_TYPE_S8:
      return DType::kInt8;
    case HB_DNN_TENSOR_TYPE_S32:
      return DType::kInt32;
    case HB_DNN_TENSOR_TYPE_F32:
      return DType::kFloat32;
    default:
      return DType::kUnsupported;
  }
}

Result<TensorShape> FromHbShape(const hbDNNTensorShape& shape) {
  if (shape.numDimensions < 0 ||
      shape.numDimensions > HB_DNN_TENSOR_MAX_DIMENSIONS) {
    return Status::InvalidArgument("invalid hbDNN tensor rank: " +
                                   std::to_string(shape.numDimensions));
  }
  TensorShape result;
  result.reserve(static_cast<size_t>(shape.numDimensions));
  for (int32_t i = 0; i < shape.numDimensions; ++i) {
    result.push_back(shape.dimensionSize[i]);
  }
  return result;
}

BpuModel::~BpuModel() {
  if (packed_handle_ != nullptr) {
    hbDNNRelease(packed_handle_);
    packed_handle_ = nullptr;
    dnn_handle_ = nullptr;
  }
}

Result<std::unique_ptr<BpuModel>> BpuModel::Load(
    std::string model_path, std::string requested_model_name) {
  if (model_path.empty()) {
    return Status::InvalidArgument("model path is empty");
  }
  if (!std::filesystem::exists(model_path)) {
    return Status::NotFound("model file not found: " + model_path);
  }

  auto model = std::unique_ptr<BpuModel>(new BpuModel());
  model->model_path_ = std::move(model_path);
  const char* files[] = {model->model_path_.c_str()};
  const int32_t rc = hbDNNInitializeFromFiles(&model->packed_handle_, files, 1);
  if (rc != HB_DNN_SUCCESS) {
    return Status::RuntimeError(HbDnnErrorMessage("hbDNNInitializeFromFiles", rc));
  }

  Status metadata_status = model->QueryMetadata(std::move(requested_model_name));
  if (!metadata_status.ok()) {
    return metadata_status;
  }
  return model;
}

Status BpuModel::QueryMetadata(std::string requested_model_name) {
  const char** model_names = nullptr;
  int32_t model_count = 0;
  int32_t rc = hbDNNGetModelNameList(&model_names, &model_count, packed_handle_);
  if (rc != HB_DNN_SUCCESS) {
    return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetModelNameList", rc));
  }
  if (model_count <= 0 || model_names == nullptr) {
    return Status::RuntimeError("hbm contains no models");
  }

  if (requested_model_name.empty()) {
    model_name_ = model_names[0];
  } else {
    bool found = false;
    for (int32_t i = 0; i < model_count; ++i) {
      if (requested_model_name == model_names[i]) {
        found = true;
        break;
      }
    }
    if (!found) {
      return Status::NotFound("model name not found in hbm: " + requested_model_name);
    }
    model_name_ = std::move(requested_model_name);
  }

  rc = hbDNNGetModelHandle(&dnn_handle_, packed_handle_, model_name_.c_str());
  if (rc != HB_DNN_SUCCESS) {
    return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetModelHandle", rc));
  }

  int32_t input_count = 0;
  rc = hbDNNGetInputCount(&input_count, dnn_handle_);
  if (rc != HB_DNN_SUCCESS) {
    return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetInputCount", rc));
  }
  inputs_.clear();
  inputs_.reserve(input_count);
  for (int32_t i = 0; i < input_count; ++i) {
    const char* name = nullptr;
    hbDNNTensorProperties properties{};
    rc = hbDNNGetInputName(&name, dnn_handle_, i);
    if (rc != HB_DNN_SUCCESS) {
      return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetInputName", rc));
    }
    rc = hbDNNGetInputTensorProperties(&properties, dnn_handle_, i);
    if (rc != HB_DNN_SUCCESS) {
      return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetInputTensorProperties", rc));
    }
    auto meta = BuildTensorMeta(name, properties);
    if (!meta.ok()) {
      return meta.status();
    }
    inputs_.push_back(meta.value());
  }

  int32_t output_count = 0;
  rc = hbDNNGetOutputCount(&output_count, dnn_handle_);
  if (rc != HB_DNN_SUCCESS) {
    return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetOutputCount", rc));
  }
  outputs_.clear();
  outputs_.reserve(output_count);
  for (int32_t i = 0; i < output_count; ++i) {
    const char* name = nullptr;
    hbDNNTensorProperties properties{};
    rc = hbDNNGetOutputName(&name, dnn_handle_, i);
    if (rc != HB_DNN_SUCCESS) {
      return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetOutputName", rc));
    }
    rc = hbDNNGetOutputTensorProperties(&properties, dnn_handle_, i);
    if (rc != HB_DNN_SUCCESS) {
      return Status::RuntimeError(HbDnnErrorMessage("hbDNNGetOutputTensorProperties", rc));
    }
    auto meta = BuildTensorMeta(name, properties);
    if (!meta.ok()) {
      return meta.status();
    }
    outputs_.push_back(meta.value());
  }

  return Status::Ok();
}

}  // namespace bpu_backend
