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

std::string ShellQuote(const std::filesystem::path& path) {
  std::string quoted = "'";
  for (char c : path.string()) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

}  // namespace

int main(int argc, char** argv) {
  EXPECT_EQ(argc, 2);
  const std::filesystem::path runner = argv[1];
  const std::filesystem::path model =
      "/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm";
  if (!std::filesystem::exists(model)) {
    std::cerr << "skipping config stdout cleanliness test; model not found: "
              << model << std::endl;
    return 0;
  }

  const auto temp_dir = std::filesystem::temp_directory_path() /
                        std::filesystem::path("bpu_runner_config_stdout_test");
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const auto stdout_path = temp_dir / "stdout.txt";
  const auto stderr_path = temp_dir / "stderr.txt";
  const auto input_path = temp_dir / "data_y.bin";
  const auto dump_dir = temp_dir / "dump";
  {
    std::ofstream input(input_path, std::ios::binary);
    std::string bytes(1 * 672 * 672 * 1, '\0');
    input.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  std::string repeated_metadata_flags;
  for (int i = 0; i < 300; ++i) {
    repeated_metadata_flags += " --print-metadata";
  }
  EXPECT_TRUE(repeated_metadata_flags.size() > 4096);

  const std::string command = ShellQuote(runner) + " --model " +
                              ShellQuote(model) + repeated_metadata_flags +
                              " --print-config-template > " +
                              ShellQuote(stdout_path) + " 2> " +
                              ShellQuote(stderr_path);
  const int exit_code = DecodeSystemStatus(std::system(command.c_str()));
  const std::string stdout_text = ReadFile(stdout_path);

  EXPECT_EQ(exit_code, 0);
  EXPECT_TRUE(stdout_text.rfind("name: \"yolov5x_bpu\"", 0) == 0);
  EXPECT_TRUE(stdout_text.find("[Horizon]") == std::string::npos);
  EXPECT_TRUE(stdout_text.find("hbDNN") == std::string::npos);
  EXPECT_TRUE(stdout_text.find("HBRT") == std::string::npos);

  std::filesystem::remove(stdout_path);
  std::filesystem::remove(stderr_path);
  const std::string mixed_command =
      ShellQuote(runner) + " --model " + ShellQuote(model) +
      " --print-config-template --input data_y=" + ShellQuote(input_path) +
      " --dump-output " + ShellQuote(dump_dir) + " > " +
      ShellQuote(stdout_path) + " 2> " + ShellQuote(stderr_path);
  const int mixed_exit_code = DecodeSystemStatus(std::system(mixed_command.c_str()));
  const std::string mixed_stdout_text = ReadFile(stdout_path);

  EXPECT_EQ(mixed_exit_code, 0);
  EXPECT_EQ(mixed_stdout_text, stdout_text);
  EXPECT_TRUE(mixed_stdout_text.find("dumped ") == std::string::npos);
  EXPECT_TRUE(mixed_stdout_text.find("[Horizon]") == std::string::npos);
  EXPECT_TRUE(mixed_stdout_text.find("hbDNN") == std::string::npos);
  EXPECT_TRUE(mixed_stdout_text.find("HBRT") == std::string::npos);
  EXPECT_TRUE(!std::filesystem::exists(dump_dir));

  std::filesystem::remove_all(temp_dir);
  return 0;
}
