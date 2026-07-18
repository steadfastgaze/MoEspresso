cmake_minimum_required(VERSION 3.27)
project(_moespresso_ds4_moe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

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

add_library(moespresso_ds4_moe)
target_sources(
  moespresso_ds4_moe
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/ds4_moe.cpp)
target_include_directories(
  moespresso_ds4_moe
  PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(moespresso_ds4_moe PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    moespresso_ds4_moe_metallib
    TITLE
    moespresso_ds4_moe
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/metal/ds4_moe.metal
    INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(moespresso_ds4_moe moespresso_ds4_moe_metallib)
endif()

nanobind_add_module(
  _moespresso_ds4_moe
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_moespresso_ds4_moe PRIVATE moespresso_ds4_moe)
target_link_options(_moespresso_ds4_moe PRIVATE -Wl,-rpath,@loader_path)
