#pragma once

#include <string>
#include <utility>

namespace bpu_backend {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kRuntimeError,
  kUnsupported,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status(); }
  static Status InvalidArgument(std::string message) {
    return Status(StatusCode::kInvalidArgument, std::move(message));
  }
  static Status NotFound(std::string message) {
    return Status(StatusCode::kNotFound, std::move(message));
  }
  static Status RuntimeError(std::string message) {
    return Status(StatusCode::kRuntimeError, std::move(message));
  }
  static Status Unsupported(std::string message) {
    return Status(StatusCode::kUnsupported, std::move(message));
  }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

template <typename T>
class Result {
 public:
  Result(T value) : status_(Status::Ok()), value_(std::move(value)) {}
  Result(Status status) : status_(std::move(status)) {}

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }
  T& value() { return value_; }
  const T& value() const { return value_; }

 private:
  Status status_;
  T value_{};
};

}  // namespace bpu_backend
