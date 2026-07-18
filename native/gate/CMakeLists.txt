cmake_minimum_required(VERSION 3.27)
project(_moespresso_gate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# NB_DOMAIN mlx + STABLE_ABI: REQUIRED so the mx.array nanobind type casters
# are shared with mlx.core (without it every call fails with a TypeError).
# nanobind version must match MLX's pin (v2.12.0 for mlx 0.31.x).
nanobind_add_module(
  _moespresso_gate
  NB_STATIC
  STABLE_ABI
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/gate.cpp)
target_link_libraries(_moespresso_gate PRIVATE mlx)
