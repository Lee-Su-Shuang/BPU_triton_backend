#pragma once

#include <memory>
#include <string>
#include <vector>

#include <hobot/dnn/hb_dnn.h>

#include "bpu_backend/bpu_status.h"
#include "bpu_backend/bpu_tensor.h"

namespace bpu_backend {

class BpuModel {
 public:
  BpuModel(const BpuModel&) = delete;
  BpuModel& operator=(const BpuModel&) = delete;
  BpuModel(BpuModel&&) = delete;
  BpuModel& operator=(BpuModel&&) = delete;
  ~BpuModel();

  static Result<std::unique_ptr<BpuModel>> Load(
      std::string model_path, std::string requested_model_name = "");

  const std::string& ModelPath() const { return model_path_; }
  const std::string& ModelName() const { return model_name_; }
  hbDNNHandle_t Handle() const { return dnn_handle_; }
  const std::vector<TensorMeta>& Inputs() const { return inputs_; }
  const std::vector<TensorMeta>& Outputs() const { return outputs_; }

 private:
  BpuModel() = default;

  Status QueryMetadata(std::string requested_model_name);

  std::string model_path_;
  std::string model_name_;
  hbDNNPackedHandle_t packed_handle_ = nullptr;
  hbDNNHandle_t dnn_handle_ = nullptr;
  std::vector<TensorMeta> inputs_;
  std::vector<TensorMeta> outputs_;
};

DType FromHbDnnType(int32_t hb_type);
Result<TensorShape> FromHbShape(const hbDNNTensorShape& shape);
std::string HbDnnErrorMessage(const std::string& operation, int32_t code);

}  // namespace bpu_backend
