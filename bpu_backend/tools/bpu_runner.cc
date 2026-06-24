#include "bpu_backend/bpu_model.h"
#include "bpu_backend/bpu_runner_args.h"
#include "bpu_backend/bpu_runtime.h"
#include "bpu_backend/bpu_tensor.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int g_pre_main_saved_stdout = -1;
bool g_pre_main_redirected_stdout = false;

bool CmdlineHasCleanStdoutMode() {
  int fd = open("/proc/self/cmdline", O_RDONLY);
  if (fd < 0) {
    return false;
  }

  std::vector<char> cmdline;
  char buffer[4096];
  while (true) {
    const ssize_t size = read(fd, buffer, sizeof(buffer));
    if (size < 0) {
      close(fd);
      return false;
    }
    if (size == 0) {
      break;
    }
    cmdline.insert(cmdline.end(), buffer, buffer + size);
  }
  close(fd);
  if (cmdline.empty()) {
    return false;
  }

  const char* flags[] = {"--print-config-template", "--print-metadata"};
  for (const char* flag : flags) {
    const size_t flag_size = std::string(flag).size();
    for (size_t offset = 0; offset + flag_size <= cmdline.size(); ++offset) {
      if (offset != 0 && cmdline[offset - 1] != '\0') {
        continue;
      }
      bool matches = true;
      for (size_t i = 0; i < flag_size; ++i) {
        if (cmdline[offset + i] != flag[i]) {
          matches = false;
          break;
        }
      }
      if (matches && (offset + flag_size == cmdline.size() ||
                      cmdline[offset + flag_size] == '\0')) {
        return true;
      }
    }
  }
  return false;
}

void PreMainRedirectStdoutForConfigTemplate() {
  if (!CmdlineHasCleanStdoutMode()) {
    return;
  }
  g_pre_main_saved_stdout = dup(STDOUT_FILENO);
  if (g_pre_main_saved_stdout < 0) {
    return;
  }
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    close(g_pre_main_saved_stdout);
    g_pre_main_saved_stdout = -1;
    return;
  }
  g_pre_main_redirected_stdout = true;
}

using PreinitFunction = void (*)();
__attribute__((section(".preinit_array"), used)) PreinitFunction
    g_preinit_redirect_stdout = PreMainRedirectStdoutForConfigTemplate;

void RestorePreMainStdout() {
  std::cout.flush();
  fflush(stdout);
  if (g_pre_main_redirected_stdout && g_pre_main_saved_stdout >= 0) {
    dup2(g_pre_main_saved_stdout, STDOUT_FILENO);
    close(g_pre_main_saved_stdout);
    g_pre_main_saved_stdout = -1;
    g_pre_main_redirected_stdout = false;
  }
}

class StdoutToStderrRedirect {
 public:
  StdoutToStderrRedirect() {
    std::cout.flush();
    fflush(stdout);
    saved_stdout_ = dup(STDOUT_FILENO);
    if (saved_stdout_ >= 0) {
      dup2(STDERR_FILENO, STDOUT_FILENO);
    }
  }

  ~StdoutToStderrRedirect() {
    std::cout.flush();
    fflush(stdout);
    if (saved_stdout_ >= 0) {
      dup2(saved_stdout_, STDOUT_FILENO);
      close(saved_stdout_);
    }
  }

 private:
  int saved_stdout_ = -1;
};

std::string SafeOutputFileName(const std::string& name) {
  std::string safe = name;
  for (char& c : safe) {
    if (c == '/' || c == '\\') {
      c = '_';
    }
  }
  return safe;
}

}  // namespace

int main(int argc, char** argv) {
  auto args = bpu_backend::ParseRunnerArgs(argc, argv);
  if (!args.ok()) {
    std::cerr << args.status().message() << std::endl;
    std::cerr << bpu_backend::RunnerUsage(argv[0]);
    return 2;
  }

  if (args.value().help) {
    RestorePreMainStdout();
    std::cout << bpu_backend::RunnerUsage(argv[0]);
    return 0;
  }

  auto model = [&args]() {
    StdoutToStderrRedirect redirect;
    return bpu_backend::BpuModel::Load(args.value().model_path);
  }();
  if (!model.ok()) {
    std::cerr << model.status().message() << std::endl;
    return 1;
  }

  std::cerr << "loaded model: " << model.value()->ModelName() << "\n";
  if (!args.value().print_config_template &&
      (args.value().print_metadata || args.value().inputs.empty())) {
    const std::string metadata = bpu_backend::MetadataTable(model.value()->Inputs(),
                                                           model.value()->Outputs());
    model.value().reset();
    RestorePreMainStdout();
    std::cout << metadata;
    return 0;
  }
  if (args.value().print_config_template) {
    auto config = bpu_backend::TritonConfigTemplate(
        "yolov5x_bpu", model.value()->Inputs(), model.value()->Outputs());
    if (!config.ok()) {
      std::cerr << config.status().message() << std::endl;
      return 1;
    }
    model.value().reset();
    RestorePreMainStdout();
    std::cout << config.value();
    return 0;
  }

  if (args.value().inputs.empty()) {
    return 0;
  }
  if (args.value().dump_output_dir.empty()) {
    std::cerr << "--dump-output is required when --input is used" << std::endl;
    return 2;
  }

  std::vector<bpu_backend::TensorBuffer> input_buffers;
  for (const auto& input_path : args.value().inputs) {
    const auto& metas = model.value()->Inputs();
    auto meta_it = std::find_if(
        metas.begin(), metas.end(),
        [&](const auto& meta) { return meta.name == input_path.name; });
    if (meta_it == metas.end()) {
      std::cerr << "input name not found in model: " << input_path.name << std::endl;
      return 2;
    }

    std::ifstream in(input_path.path, std::ios::binary);
    if (!in) {
      std::cerr << "failed to open input file: " << input_path.path << std::endl;
      return 2;
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    bpu_backend::TensorBuffer buffer;
    buffer.meta = *meta_it;
    buffer.bytes = std::move(bytes);
    auto size_status =
        bpu_backend::ValidateTensorByteSize(buffer.meta, buffer.bytes.size());
    if (!size_status.ok()) {
      std::cerr << size_status.message() << std::endl;
      return 2;
    }
    input_buffers.push_back(std::move(buffer));
  }

  std::error_code mkdir_error;
  std::filesystem::create_directories(args.value().dump_output_dir, mkdir_error);
  if (mkdir_error) {
    std::cerr << "failed to create output directory: "
              << args.value().dump_output_dir << ": " << mkdir_error.message()
              << std::endl;
    return 2;
  }
  bpu_backend::BpuRuntime runtime(*model.value());
  auto outputs = runtime.Infer(input_buffers);
  if (!outputs.ok()) {
    std::cerr << outputs.status().message() << std::endl;
    return 1;
  }

  std::set<std::string> output_file_names;
  for (const auto& output : outputs.value()) {
    const std::string file_name = SafeOutputFileName(output.meta.name) + ".bin";
    if (!output_file_names.insert(file_name).second) {
      std::cerr << "output file name collision after sanitizing: "
                << output.meta.name << std::endl;
      return 2;
    }
    const std::filesystem::path path =
        std::filesystem::path(args.value().dump_output_dir) / file_name;
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      std::cerr << "failed to open output file: " << path.string() << std::endl;
      return 2;
    }
    out.write(reinterpret_cast<const char*>(output.bytes.data()),
              static_cast<std::streamsize>(output.bytes.size()));
    if (!out) {
      std::cerr << "failed to write output file: " << path.string() << std::endl;
      return 2;
    }
    std::cout << "dumped " << path.string() << " bytes=" << output.bytes.size()
              << "\n";
  }
  return 0;
}
