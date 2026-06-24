#pragma once

#include <memory>
#include <vector>

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp_sys.h>

#include "bpu_backend/bpu_model.h"
#include "bpu_backend/bpu_status.h"
#include "bpu_backend/bpu_tensor.h"

namespace bpu_backend {

class BpuRuntime {
 public:
  explicit BpuRuntime(const BpuModel& model) : model_(model) {}

  Result<std::vector<TensorBuffer>> Infer(
      const std::vector<TensorBuffer>& inputs) const;

 private:
  const BpuModel& model_;
};

class UcpTensorBuffer {
 public:
  UcpTensorBuffer() = default;
  UcpTensorBuffer(const UcpTensorBuffer&) = delete;
  UcpTensorBuffer& operator=(const UcpTensorBuffer&) = delete;
  UcpTensorBuffer(UcpTensorBuffer&& other) noexcept;
  UcpTensorBuffer& operator=(UcpTensorBuffer&& other) noexcept;
  ~UcpTensorBuffer();

  Status Allocate(const TensorMeta& meta);
  hbDNNTensor* tensor() { return &tensor_; }
  const hbDNNTensor* tensor() const { return &tensor_; }
  void* data() { return tensor_.sysMem.virAddr; }
  const void* data() const { return tensor_.sysMem.virAddr; }
  size_t size() const { return static_cast<size_t>(tensor_.sysMem.memSize); }
  size_t allocated_size() const { return allocated_size_; }

 private:
  void Release();
  hbDNNTensor tensor_{};
  size_t allocated_size_ = 0;
  bool allocated_ = false;
};

}  // namespace bpu_backend
