cmake_minimum_required(VERSION 3.20)
project(scaled_dot_product_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(attention_lib STATIC)

target_sources(attention_lib PRIVATE
        src/matrix_multiplication.cpp
)

target_include_directories(attention_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(attention::core ALIAS attention_lib)

enable_testing()
add_subdirectory(tests)