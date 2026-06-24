#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>

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

namespace {

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

int DecodeSystemStatus(int status) {
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  EXPECT_EQ(argc, 2);
  const std::filesystem::path runner = argv[1];
  const std::filesystem::path model =
      "/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm";
  if (!std::filesystem::exists(model)) {
    std::cerr << "skipping runner behavior test; model not found: " << model
              << std::endl;
    return 0;
  }

  const auto temp_dir = std::filesystem::temp_directory_path() /
                        std::filesystem::path("bpu_runner_behavior_test");
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const auto input_path = temp_dir / "data_y.bin";
  std::ofstream input(input_path, std::ios::binary);
  std::string bytes(1 * 672 * 672 * 1, '\0');
  input.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  input.close();

  const auto output_path = temp_dir / "stderr.txt";
  const std::string command = runner.string() + " --model " + model.string() +
                              " --input data_y=" + input_path.string() +
                              " > " + output_path.string() + " 2>&1";
  const int exit_code = DecodeSystemStatus(std::system(command.c_str()));
  const std::string output = ReadFile(output_path);

  EXPECT_EQ(exit_code, 2);
  EXPECT_TRUE(output.find("--dump-output is required when --input is used") !=
              std::string::npos);

  std::filesystem::remove_all(temp_dir);
  return 0;
}
