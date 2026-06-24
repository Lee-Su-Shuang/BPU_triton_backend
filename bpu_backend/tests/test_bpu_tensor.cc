#include "bpu_backend/bpu_tensor.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#define EXPECT_EQ(actual, expected)                                            \
  do {                                                                         \
    auto actual_value = (actual);                                              \
    auto expected_value = (expected);                                          \
    if (actual_value != expected_value) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " expected "                \
                << #actual << " == " << #expected << ", got " << actual_value  \
                << " vs " << expected_value << std::endl;                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  using namespace bpu_backend;

  EXPECT_EQ(ElementSizeBytes(DType::kUInt8), static_cast<size_t>(1));
  EXPECT_EQ(ElementSizeBytes(DType::kInt32), static_cast<size_t>(4));
  EXPECT_EQ(ShapeElementCount({1, 672, 672, 1}), static_cast<int64_t>(451584));
  EXPECT_EQ(TensorByteSize(DType::kUInt8, {1, 672, 672, 1}),
            static_cast<size_t>(451584));
  EXPECT_EQ(TensorByteSize(DType::kUInt8, {1, 336, 336, 2}),
            static_cast<size_t>(225792));
  EXPECT_EQ(TensorByteSize(DType::kInt32, {1, 84, 84, 255}),
            static_cast<size_t>(7197120));
  EXPECT_EQ(TensorByteSize(DType::kInt32, {1, 42, 42, 255}),
            static_cast<size_t>(1799280));
  EXPECT_EQ(TensorByteSize(DType::kInt32, {1, 21, 21, 255}),
            static_cast<size_t>(449820));
  EXPECT_EQ(TensorByteSize(DType::kInt32,
                                 {std::numeric_limits<int64_t>::max() / 2 + 2}),
            static_cast<size_t>(0));
  EXPECT_EQ(DTypeName(DType::kUInt8), std::string("UINT8"));
  EXPECT_EQ(DTypeName(DType::kInt32), std::string("INT32"));
  EXPECT_EQ(ShapeToString({1, 672, 672, 1}), std::string("[1,672,672,1]"));

  return 0;
}
