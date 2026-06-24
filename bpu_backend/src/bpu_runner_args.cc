#include "bpu_backend/bpu_runner_args.h"

#include <sstream>

namespace bpu_backend {
namespace {

bool MissingOptionValue(int argc, char** argv, int index) {
  return index + 1 >= argc || std::string(argv[index + 1]).rfind("--", 0) == 0;
}

Result<NamedInputPath> ParseNamedInput(const std::string& value) {
  const size_t pos = value.find('=');
  if (pos == std::string::npos || pos == 0 || pos + 1 >= value.size()) {
    return Status::InvalidArgument("--input must use name=path format: " + value);
  }
  return NamedInputPath{value.substr(0, pos), value.substr(pos + 1)};
}

}  // namespace

std::string RunnerUsage(const char* argv0) {
  std::ostringstream out;
  out << "Usage: " << argv0 << " --model model.hbm "
      << "[--input name=file.bin ...] "
      << "[--dump-output output_dir] "
      << "[--print-metadata] [--print-config-template]\n";
  return out.str();
}

Result<RunnerArgs> ParseRunnerArgs(int argc, char** argv) {
  RunnerArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--model") {
      if (MissingOptionValue(argc, argv, i)) {
        return Status::InvalidArgument("--model requires a value");
      }
      args.model_path = argv[++i];
    } else if (key == "--input") {
      if (MissingOptionValue(argc, argv, i)) {
        return Status::InvalidArgument("--input requires a value");
      }
      auto parsed = ParseNamedInput(argv[++i]);
      if (!parsed.ok()) {
        return parsed.status();
      }
      args.inputs.push_back(parsed.value());
    } else if (key == "--dump-output") {
      if (MissingOptionValue(argc, argv, i)) {
        return Status::InvalidArgument("--dump-output requires a value");
      }
      args.dump_output_dir = argv[++i];
    } else if (key == "--print-metadata") {
      args.print_metadata = true;
    } else if (key == "--print-config-template") {
      args.print_config_template = true;
    } else if (key == "--help" || key == "-h") {
      args.help = true;
    } else {
      return Status::InvalidArgument("unknown argument: " + key);
    }
  }

  if (!args.help && args.model_path.empty()) {
    return Status::InvalidArgument("missing required --model");
  }
  return args;
}

}  // namespace bpu_backend
