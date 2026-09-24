cmake_minimum_required(VERSION 3.27)
project(moespresso_native LANGUAGES CXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(FATAL_ERROR "MoEspresso native extensions require arm64 macOS.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_VARIABLE MLX_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(MLX CONFIG REQUIRED)

# Some command-line tool installations expose libc++ headers only in the SDK.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <cstddef>\nint main() { return 0; }" HAS_LIBCPP_HEADERS)
if(NOT HAS_LIBCPP_HEADERS)
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE SDK_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  add_compile_options(-nostdinc++ "SHELL:-isystem \"${SDK_ROOT}/usr/include/c++/v1\"")
endif()

# Install only extension modules, resolving the separately installed MLX library.
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "@loader_path/../../mlx/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# Keep source and isolated-build paths out of compiled diagnostics.
get_filename_component(NANOBIND_PACKAGE_DIR "${nanobind_ROOT}/.." ABSOLUTE)
get_filename_component(MLX_PACKAGE_DIR "${MLX_ROOT}/../../.." ABSOLUTE)
add_compile_options(
  "-ffile-prefix-map=${PROJECT_SOURCE_DIR}=/moespresso"
  "-ffile-prefix-map=${PROJECT_BINARY_DIR}=/moespresso-build"
  "-ffile-prefix-map=${NANOBIND_PACKAGE_DIR}=/nanobind"
  "-ffile-prefix-map=${MLX_PACKAGE_DIR}=/mlx")

add_subdirectory(native/gate)
install(TARGETS _moespresso_gate
  LIBRARY DESTINATION moespresso/_native)
