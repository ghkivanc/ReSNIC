cmake_minimum_required(VERSION 3.18)
project(snic LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# -----------------------------
# SNIC core library
# -----------------------------
add_library(snic_lib STATIC
    snic.cpp
    snic_kernels.cu
)

target_compile_options(snic_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_compile_options(snic_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
)




target_include_directories(snic_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(snic_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# -----------------------------
# Python extension
# -----------------------------
pybind11_add_module(snic_ext bindings.cpp)

target_link_libraries(snic_ext
    PRIVATE snic_lib
    PRIVATE CUDA::cudart
)

target_compile_features(snic_ext PRIVATE cxx_std_17)


