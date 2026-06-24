#include "bpu_backend/bpu_tensor.h"

#include <iostream>
#include <string>
#include <vector>

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

#define EXPECT_CONTAINS(text, needle)                                          \
  do {                                                                         \
    if ((text).find(needle) == std::string::npos) {                            \
      std::cerr << __FILE__ << ":" << __LINE__ << " missing substring: "       \
                << needle << "\nText:\n" << text << std::endl;                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  using namespace bpu_backend;
  std::vector<TensorMeta> inputs = {
      TensorMeta{"data_y", DType::kUInt8, {1, 672, 672, 1}, -1},
      TensorMeta{"data_uv", DType::kUInt8, {1, 336, 336, 2}, -1},
  };
  std::vector<TensorMeta> outputs = {
      TensorMeta{"output", DType::kInt32, {1, 84, 84, 255}, 7225344},
      TensorMeta{"1310", DType::kInt32, {1, 42, 42, 255}, 1806336},
      TensorMeta{"1312", DType::kInt32, {1, 21, 21, 255}, 451584},
  };

  const auto config = TritonConfigTemplate("yolov5x_bpu", inputs, outputs);
  EXPECT_TRUE(config.ok());
  EXPECT_CONTAINS(config.value(), "name: \"yolov5x_bpu\"");
  EXPECT_CONTAINS(config.value(), "backend: \"bpu\"");
  EXPECT_CONTAINS(config.value(), "max_batch_size: 0");
  EXPECT_CONTAINS(config.value(), "name: \"data_y\"");
  EXPECT_CONTAINS(config.value(), "data_type: TYPE_UINT8");
  EXPECT_CONTAINS(config.value(), "dims: [1, 672, 672, 1]");
  EXPECT_CONTAINS(config.value(), "name: \"output\"");
  EXPECT_CONTAINS(config.value(), "data_type: TYPE_INT32");
  EXPECT_CONTAINS(config.value(), "dims: [1, 84, 84, 255]");

  outputs[0].dtype = DType::kUnsupported;
  const auto unsupported_config = TritonConfigTemplate("bad", inputs, outputs);
  EXPECT_TRUE(!unsupported_config.ok());
  EXPECT_EQ(static_cast<int>(unsupported_config.status().code()),
            static_cast<int>(StatusCode::kUnsupported));
  EXPECT_EQ(unsupported_config.status().message(),
            std::string("unsupported tensor dtype for output 'output'"));

  return 0;
}
