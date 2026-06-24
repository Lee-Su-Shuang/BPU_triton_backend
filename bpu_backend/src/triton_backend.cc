#include "bpu_backend/bpu_model.h"
#include "bpu_backend/bpu_runtime.h"

#include "triton/core/tritonbackend.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef RETURN_IF_ERROR
#define RETURN_IF_ERROR(X)           \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      return err__;                  \
    }                                \
  } while (false)
#endif

namespace {

TRITONSERVER_Error* ErrorNew(TRITONSERVER_Error_Code code,
                             const std::string& message) {
  return TRITONSERVER_ErrorNew(code, message.c_str());
}

TRITONSERVER_Error_Code TritonErrorCode(const bpu_backend::Status& status) {
  switch (status.code()) {
    case bpu_backend::StatusCode::kOk:
      return TRITONSERVER_ERROR_UNKNOWN;
    case bpu_backend::StatusCode::kInvalidArgument:
      return TRITONSERVER_ERROR_INVALID_ARG;
    case bpu_backend::StatusCode::kNotFound:
      return TRITONSERVER_ERROR_NOT_FOUND;
    case bpu_backend::StatusCode::kRuntimeError:
      return TRITONSERVER_ERROR_INTERNAL;
    case bpu_backend::StatusCode::kUnsupported:
      return TRITONSERVER_ERROR_UNSUPPORTED;
  }
  return TRITONSERVER_ERROR_INTERNAL;
}

const bpu_backend::TensorMeta* FindMeta(
    const std::vector<bpu_backend::TensorMeta>& metas,
    const std::string& name) {
  for (const auto& meta : metas) {
    if (meta.name == name) {
      return &meta;
    }
  }
  return nullptr;
}

TRITONSERVER_DataType TritonDataType(bpu_backend::DType dtype) {
  switch (dtype) {
    case bpu_backend::DType::kUInt8:
      return TRITONSERVER_TYPE_UINT8;
    case bpu_backend::DType::kInt8:
      return TRITONSERVER_TYPE_INT8;
    case bpu_backend::DType::kInt32:
      return TRITONSERVER_TYPE_INT32;
    case bpu_backend::DType::kFloat32:
      return TRITONSERVER_TYPE_FP32;
    case bpu_backend::DType::kUnsupported:
      return TRITONSERVER_TYPE_INVALID;
  }
  return TRITONSERVER_TYPE_INVALID;
}

TRITONSERVER_Error* CollectInput(TRITONBACKEND_Request* request,
                                 const bpu_backend::TensorMeta& meta,
                                 bpu_backend::TensorBuffer* buffer) {
  TRITONBACKEND_Input* input = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, meta.name.c_str(), &input));

  const char* input_name = nullptr;
  TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
  const int64_t* shape = nullptr;
  uint32_t dims_count = 0;
  uint64_t byte_size = 0;
  uint32_t buffer_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, &input_name, &datatype, &shape, &dims_count, &byte_size,
      &buffer_count));

  const size_t expected = bpu_backend::TensorByteSize(meta.dtype, meta.shape);
  if (byte_size != expected) {
    return ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                    "input byte size mismatch for " + meta.name);
  }

  buffer->meta = meta;
  buffer->bytes.resize(expected);
  size_t copied = 0;
  for (uint32_t idx = 0; idx < buffer_count; ++idx) {
    const void* content = nullptr;
    uint64_t content_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, idx, &content, &content_byte_size, &memory_type,
        &memory_type_id));
    if (memory_type != TRITONSERVER_MEMORY_CPU) {
      return ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                      "non-CPU input buffer for " + meta.name);
    }
    if (copied + content_byte_size > buffer->bytes.size()) {
      return ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                      "input buffer overflow for " + meta.name);
    }
    if (content_byte_size != 0 && content == nullptr) {
      return ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                      "null input buffer for " + meta.name);
    }
    std::memcpy(buffer->bytes.data() + copied, content, content_byte_size);
    copied += content_byte_size;
  }
  if (copied != expected) {
    return ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                    "input byte size mismatch for " + meta.name);
  }
  return nullptr;
}

TRITONSERVER_Error* SendAndDeleteError(TRITONBACKEND_Response* response,
                                       TRITONSERVER_Error* error) {
  TRITONSERVER_Error* send_error = TRITONBACKEND_ResponseSend(
      response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error);
  TRITONSERVER_ErrorDelete(error);
  return send_error;
}

struct ModelState {
  std::unique_ptr<bpu_backend::BpuModel> model;
};

}  // namespace

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  const char* name = nullptr;
  TRITONBACKEND_BackendName(backend, &name);
  uint32_t api_major = 0;
  uint32_t api_minor = 0;
  TRITONBACKEND_ApiVersion(&api_major, &api_minor);
  if (api_major != TRITONBACKEND_API_VERSION_MAJOR ||
      api_minor < TRITONBACKEND_API_VERSION_MINOR) {
    return ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                    "Triton backend API version is incompatible with bpu backend");
  }
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* repository_path = nullptr;
  TRITONBACKEND_ArtifactType artifact_type;
  const char* model_name = nullptr;
  uint64_t version = 0;
  TRITONSERVER_Error* error = nullptr;
  error = TRITONBACKEND_ModelRepository(model, &artifact_type, &repository_path);
  if (error != nullptr) {
    return error;
  }
  error = TRITONBACKEND_ModelName(model, &model_name);
  if (error != nullptr) {
    return error;
  }
  error = TRITONBACKEND_ModelVersion(model, &version);
  if (error != nullptr) {
    return error;
  }

  const std::string hbm_path = std::string(repository_path) + "/" +
                               std::to_string(version) + "/model.hbm";
  auto loaded = bpu_backend::BpuModel::Load(hbm_path);
  if (!loaded.ok()) {
    return ErrorNew(TritonErrorCode(loaded.status()), loaded.status().message());
  }

  auto* state = new ModelState();
  state->model = std::move(loaded.value());
  TRITONSERVER_Error* set_state_error =
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(state));
  if (set_state_error != nullptr) {
    delete state;
    return set_state_error;
  }
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* raw_state = nullptr;
  TRITONBACKEND_ModelState(model, &raw_state);
  delete reinterpret_cast<ModelState*>(raw_state);
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  TRITONBACKEND_Model* triton_model = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &triton_model));
  void* raw_state = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(triton_model, &raw_state));
  auto* state = reinterpret_cast<ModelState*>(raw_state);
  bpu_backend::BpuRuntime runtime(*state->model);

  for (uint32_t i = 0; i < request_count; ++i) {
    TRITONBACKEND_Response* response = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, requests[i]));

    std::vector<bpu_backend::TensorBuffer> inputs;
    bool failed = false;
    for (const auto& meta : state->model->Inputs()) {
      bpu_backend::TensorBuffer input;
      TRITONSERVER_Error* err = CollectInput(requests[i], meta, &input);
      if (err != nullptr) {
        RETURN_IF_ERROR(SendAndDeleteError(response, err));
        failed = true;
        break;
      }
      inputs.push_back(std::move(input));
    }

    if (!failed) {
      auto outputs = runtime.Infer(inputs);
      if (!outputs.ok()) {
        RETURN_IF_ERROR(SendAndDeleteError(
            response, ErrorNew(TritonErrorCode(outputs.status()),
                               outputs.status().message())));
      } else {
        for (const auto& output : outputs.value()) {
          TRITONBACKEND_Output* triton_output = nullptr;
          TRITONSERVER_Error* err = TRITONBACKEND_ResponseOutput(
              response, &triton_output, output.meta.name.c_str(),
              TritonDataType(output.meta.dtype), output.meta.shape.data(),
              static_cast<uint32_t>(output.meta.shape.size()));
          if (err != nullptr) {
            RETURN_IF_ERROR(SendAndDeleteError(response, err));
            failed = true;
            break;
          }
          void* buffer = nullptr;
          TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
          int64_t memory_type_id = 0;
          err = TRITONBACKEND_OutputBuffer(triton_output, &buffer,
                                           output.bytes.size(), &memory_type,
                                           &memory_type_id);
          if (err != nullptr) {
            RETURN_IF_ERROR(SendAndDeleteError(response, err));
            failed = true;
            break;
          }
          if (memory_type != TRITONSERVER_MEMORY_CPU) {
            RETURN_IF_ERROR(SendAndDeleteError(
                response, ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                                   "non-CPU output buffer for " +
                                       output.meta.name)));
            failed = true;
            break;
          }
          std::memcpy(buffer, output.bytes.data(), output.bytes.size());
        }
        if (!failed) {
          RETURN_IF_ERROR(TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr));
        }
      }
    }

    RETURN_IF_ERROR(
        TRITONBACKEND_RequestRelease(requests[i], TRITONSERVER_REQUEST_RELEASE_ALL));
  }
  return nullptr;
}

}  // extern "C"
