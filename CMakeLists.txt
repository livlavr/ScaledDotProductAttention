cmake_minimum_required(VERSION 3.20)
project(scaled_dot_product_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -funroll-loops)
        add_definitions(-DNDEBUG)
endif()

add_library(attention_lib STATIC src/matrix_multiplication.cpp)
add_library(attention::core ALIAS attention_lib)

target_include_directories(attention_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(benchmark_run benchmarks/benchmark.cpp)
target_link_libraries(benchmark_run PRIVATE attention::core)

enable_testing()
add_subdirectory(tests)