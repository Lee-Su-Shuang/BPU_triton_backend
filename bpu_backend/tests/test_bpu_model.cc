#include "bpu_backend/bpu_model.h"

#include <iostream>
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

  hbDNNTensorShape valid{};
  valid.numDimensions = 4;
  valid.dimensionSize[0] = 1;
  valid.dimensionSize[1] = 672;
  valid.dimensionSize[2] = 672;
  valid.dimensionSize[3] = 1;
  auto valid_shape = FromHbShape(valid);
  EXPECT_TRUE(valid_shape.ok());
  EXPECT_EQ(valid_shape.value().size(), static_cast<size_t>(4));
  EXPECT_EQ(valid_shape.value()[1], static_cast<int64_t>(672));

  hbDNNTensorShape negative{};
  negative.numDimensions = -1;
  auto negative_shape = FromHbShape(negative);
  EXPECT_TRUE(!negative_shape.ok());
  EXPECT_EQ(static_cast<int>(negative_shape.status().code()),
            static_cast<int>(StatusCode::kInvalidArgument));
  EXPECT_EQ(negative_shape.status().message(),
            std::string("invalid hbDNN tensor rank: -1"));

  hbDNNTensorShape oversized{};
  oversized.numDimensions = HB_DNN_TENSOR_MAX_DIMENSIONS + 1;
  auto oversized_shape = FromHbShape(oversized);
  EXPECT_TRUE(!oversized_shape.ok());
  EXPECT_EQ(static_cast<int>(oversized_shape.status().code()),
            static_cast<int>(StatusCode::kInvalidArgument));
  EXPECT_EQ(oversized_shape.status().message(),
            std::string("invalid hbDNN tensor rank: ") +
                std::to_string(HB_DNN_TENSOR_MAX_DIMENSIONS + 1));

  return 0;
}
