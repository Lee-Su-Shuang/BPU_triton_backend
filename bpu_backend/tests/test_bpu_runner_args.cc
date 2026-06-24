#include "bpu_backend/bpu_runner_args.h"

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

  const char* argv_ok[] = {
      "bpu_runner",
      "--model", "/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm",
      "--input", "data_y=./data_y.bin",
      "--input", "data_uv=./data_uv.bin",
      "--dump-output", "./outputs",
      "--print-metadata"};
  auto ok = ParseRunnerArgs(10, const_cast<char**>(argv_ok));
  EXPECT_TRUE(ok.ok());
  EXPECT_EQ(ok.value().model_path,
            std::string("/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm"));
  EXPECT_EQ(ok.value().inputs.size(), static_cast<size_t>(2));
  EXPECT_EQ(ok.value().inputs[0].name, std::string("data_y"));
  EXPECT_EQ(ok.value().inputs[0].path, std::string("./data_y.bin"));
  EXPECT_EQ(ok.value().dump_output_dir, std::string("./outputs"));
  EXPECT_TRUE(ok.value().print_metadata);

  const char* argv_missing_model[] = {"bpu_runner", "--input", "data_y=x.bin"};
  auto missing = ParseRunnerArgs(3, const_cast<char**>(argv_missing_model));
  EXPECT_TRUE(!missing.ok());
  EXPECT_EQ(missing.status().message(), std::string("missing required --model"));

  const char* argv_bad_input[] = {"bpu_runner", "--model", "m.hbm", "--input", "data_y"};
  auto bad_input = ParseRunnerArgs(5, const_cast<char**>(argv_bad_input));
  EXPECT_TRUE(!bad_input.ok());
  EXPECT_EQ(bad_input.status().message(),
            std::string("--input must use name=path format: data_y"));

  const char* argv_model_flag_value[] = {"bpu_runner", "--model", "--print-metadata"};
  auto model_flag_value = ParseRunnerArgs(3, const_cast<char**>(argv_model_flag_value));
  EXPECT_TRUE(!model_flag_value.ok());
  EXPECT_EQ(model_flag_value.status().message(),
            std::string("--model requires a value"));

  const char* argv_input_flag_value[] = {"bpu_runner", "--model", "m.hbm", "--input", "--print-metadata"};
  auto input_flag_value = ParseRunnerArgs(5, const_cast<char**>(argv_input_flag_value));
  EXPECT_TRUE(!input_flag_value.ok());
  EXPECT_EQ(input_flag_value.status().message(),
            std::string("--input requires a value"));

  const char* argv_dump_flag_value[] = {"bpu_runner", "--model", "m.hbm", "--dump-output", "--print-metadata"};
  auto dump_flag_value = ParseRunnerArgs(5, const_cast<char**>(argv_dump_flag_value));
  EXPECT_TRUE(!dump_flag_value.ok());
  EXPECT_EQ(dump_flag_value.status().message(),
            std::string("--dump-output requires a value"));

  const char* argv_help[] = {"bpu_runner", "--help"};
  auto help = ParseRunnerArgs(2, const_cast<char**>(argv_help));
  EXPECT_TRUE(help.ok());
  EXPECT_TRUE(help.value().help);

  return 0;
}
