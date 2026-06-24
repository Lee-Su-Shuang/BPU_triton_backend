#include "bpu_backend/bpu_runtime.h"

#include <iostream>
#include <limits>
#include <string>

#define EXPECT_TRUE(value)                                                     \
  do {                                                                         \
    if (!(value)) {                                                            \
      std::cerr << __FILE__ << ":" << __LINE__ << " expected true: "          \
                << #value << std::endl;                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_EQ(actual, expected)                                            \
  do {                                                                         \
    auto actual_value = (actual);                                              \
    auto expected_value = (expected);                                          \
    if (actual_value != expected_value) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " expected " << #actual     \
                << " == " << #expected << ", got '" << actual_value           \
                << "' vs '" << expected_value << "'" << std::endl;           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  using namespace bpu_backend;

  TensorMeta meta;
  meta.name = "data_y";
  meta.dtype = DType::kUInt8;
  meta.shape = {1, 672, 672, 1};
  meta.aligned_byte_size = 1;
  meta.strides = {451584, 672, 1, 1};

  UcpTensorBuffer tensor;
  Status status = tensor.Allocate(meta);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(tensor.tensor()->properties.stride[0], static_cast<int64_t>(451584));
  EXPECT_EQ(tensor.tensor()->properties.stride[1], static_cast<int64_t>(672));
  EXPECT_EQ(tensor.tensor()->properties.stride[2], static_cast<int64_t>(1));
  EXPECT_EQ(tensor.tensor()->properties.stride[3], static_cast<int64_t>(1));
  EXPECT_TRUE(tensor.allocated_size() >= static_cast<size_t>(451584));

  TensorMeta padded;
  padded.name = "padded";
  padded.dtype = DType::kUInt8;
  padded.shape = {1, 2, 3, 1};
  padded.aligned_byte_size = 1;
  padded.strides = {0, 16, 4, 0};

  UcpTensorBuffer padded_tensor;
  Status padded_status = padded_tensor.Allocate(padded);
  EXPECT_TRUE(padded_status.ok());
  EXPECT_TRUE(padded_tensor.allocated_size() >= static_cast<size_t>(25));

  TensorMeta invalid_stride;
  invalid_stride.name = "invalid_stride";
  invalid_stride.dtype = DType::kUInt8;
  invalid_stride.shape = {1, 2, 3, 1};
  invalid_stride.aligned_byte_size = 1;
  invalid_stride.strides = {0, std::numeric_limits<int64_t>::max(), 4, 0};

  UcpTensorBuffer invalid_tensor;
  Status invalid_status = invalid_tensor.Allocate(invalid_stride);
  EXPECT_TRUE(!invalid_status.ok());

  return 0;
}
