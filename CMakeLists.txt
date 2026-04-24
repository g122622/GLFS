cmake_minimum_required(VERSION 3.20)

project(GPULearnedFS VERSION 0.1.0 LANGUAGES CXX)

option(GLFS_BUILD_TESTS "Build the GLFS self-tests" ON)
option(GLFS_BUILD_BENCHMARKS "Build benchmark helpers" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE3 REQUIRED fuse3)

add_library(glfs_core
    src/utils/json_parser.cpp
    src/utils/perfetto_integration.cpp
    src/config/config_manager.cpp
    src/core/path_encoder.cpp
    src/core/gpu_index_adapter.cpp
    src/core/backend_common.cpp
    src/core/backends/rmi_cuda/backend.cpp
    src/core/backends/g_index/backend.cpp
    src/core/backends/cu_learned/backend.cpp
    src/core/backends/pgm_gpu/backend.cpp
    src/core/backends/ligpu/backend.cpp
    src/fuse/gpufs_ops.cpp
    src/fuse/client_api.cpp
    src/benchmark/runner.cpp
)

target_include_directories(glfs_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${FUSE3_INCLUDE_DIRS}
)

target_compile_definitions(glfs_core PUBLIC
    GLFS_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if (MSVC)
    target_compile_options(glfs_core PRIVATE /W4)
else()
    target_compile_options(glfs_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_options(glfs_core PUBLIC ${FUSE3_CFLAGS_OTHER})
target_link_directories(glfs_core PUBLIC ${FUSE3_LIBRARY_DIRS})
target_link_libraries(glfs_core PUBLIC ${FUSE3_LIBRARIES})

add_executable(gpufs src/main.cpp)
target_link_libraries(gpufs PRIVATE glfs_core)

if (GLFS_BUILD_TESTS)
    enable_testing()
    add_executable(glfs_tests tests/test_main.cpp)
    target_link_libraries(glfs_tests PRIVATE glfs_core)
    add_test(NAME glfs_tests COMMAND glfs_tests)
endif()

install(TARGETS gpufs DESTINATION bin)
