#pragma once

#include <string>
#include <vector>

#include "bpu_backend/bpu_status.h"

namespace bpu_backend {

struct NamedInputPath {
  std::string name;
  std::string path;
};

struct RunnerArgs {
  std::string model_path;
  std::vector<NamedInputPath> inputs;
  std::string dump_output_dir;
  bool print_metadata = false;
  bool print_config_template = false;
  bool help = false;
};

Result<RunnerArgs> ParseRunnerArgs(int argc, char** argv);
std::string RunnerUsage(const char* argv0);

}  // namespace bpu_backend
