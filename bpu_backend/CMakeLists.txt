cmake_minimum_required(VERSION 3.22)
project(bpu_triton_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BPU_BACKEND_BUILD_TESTS "Build BPU backend tests" ON)
option(BPU_BACKEND_BUILD_TRITON "Build libtriton_bpu.so" OFF)

set(BPU_BACKEND_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(TRITON_BACKEND_REPO
    "/home/sunrise/triton_backend_BPU/deps/triton_core"
    CACHE PATH "Path to the Triton backend/core repository")

add_library(bpu_core
  src/bpu_model.cc
  src/bpu_runner_args.cc
  src/bpu_runtime.cc
  src/bpu_tensor.cc
)

target_include_directories(bpu_core
  PUBLIC
    ${BPU_BACKEND_ROOT}/include
  PRIVATE
    /usr/include
    /usr/include/hobot
)

target_compile_options(bpu_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
set_target_properties(bpu_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_directories(bpu_core PUBLIC /usr/hobot/lib /usr/lib /usr/lib/aarch64-linux-gnu)
target_link_libraries(bpu_core PUBLIC dnn hbrt4 hbucp)

add_executable(bpu_runner tools/bpu_runner.cc)
target_link_libraries(bpu_runner PRIVATE bpu_core)

if(BPU_BACKEND_BUILD_TESTS)
  enable_testing()
  add_executable(test_bpu_tensor tests/test_bpu_tensor.cc)
  target_link_libraries(test_bpu_tensor PRIVATE bpu_core)
  add_test(NAME test_bpu_tensor COMMAND test_bpu_tensor)

  add_executable(test_bpu_runner_args tests/test_bpu_runner_args.cc)
  target_link_libraries(test_bpu_runner_args PRIVATE bpu_core)
  add_test(NAME test_bpu_runner_args COMMAND test_bpu_runner_args)

  add_executable(test_config_template tests/test_config_template.cc)
  target_link_libraries(test_config_template PRIVATE bpu_core)
  add_test(NAME test_config_template COMMAND test_config_template)

  add_executable(test_bpu_model tests/test_bpu_model.cc)
  target_link_libraries(test_bpu_model PRIVATE bpu_core)
  add_test(NAME test_bpu_model COMMAND test_bpu_model)

  add_executable(test_bpu_runtime tests/test_bpu_runtime.cc)
  target_link_libraries(test_bpu_runtime PRIVATE bpu_core)
  add_test(NAME test_bpu_runtime COMMAND test_bpu_runtime)

  add_executable(test_bpu_runner_behavior tests/test_bpu_runner_behavior.cc)
  add_test(NAME test_bpu_runner_behavior COMMAND test_bpu_runner_behavior $<TARGET_FILE:bpu_runner>)

  add_executable(test_bpu_runner_config_stdout tests/test_bpu_runner_config_stdout.cc)
  add_test(NAME test_bpu_runner_config_stdout COMMAND test_bpu_runner_config_stdout $<TARGET_FILE:bpu_runner>)
endif()


if(BPU_BACKEND_BUILD_TRITON)
  add_library(triton_bpu SHARED src/triton_backend.cc)
  target_include_directories(triton_bpu PRIVATE
    ${TRITON_BACKEND_REPO}/include
    ${BPU_BACKEND_ROOT}/include
  )
  target_link_libraries(triton_bpu PRIVATE bpu_core)
  set_target_properties(triton_bpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_bpu
  )
endif()
