# Minimum required version of CMake
cmake_minimum_required(VERSION 3.18)

# Name of the project
project(TRTReceiver LANGUAGES CXX CUDA)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# Point CMake to your TensorRT install
set(TensorRT_DIR "C:/Users/singh/TensorRT-10.13.2.6.Windows.win10.cuda-12.9/TensorRT-10.13.2.6")

# Add TensorRT headers
include_directories(${TensorRT_DIR}/include)

# Add TensorRT libraries
link_directories(${TensorRT_DIR}/lib)

# Add the source directory to the include path
include_directories(runtime)

# Set CUDA toolset explicitly
set(CMAKE_GENERATOR_TOOLSET "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")

# Define a shared library named 'trt_receiver' from the source files
cuda_add_library(trt_receiver SHARED
    runtime/trt_receiver.cpp
    runtime/data_processing.cu
)

# Link TensorRT libraries (plain signature, must match cuda_add_library style)
target_link_libraries(trt_receiver
    nvinfer
    nvonnxparser
    nvparsers
    nvinfer_plugin
)

# --- Optional Python Bindings with Nanobind ---
option(ENABLE_NANOBIND "Enable nanobind Python bindings" OFF)
if(ENABLE_NANOBIND)
    find_package(nanobind REQUIRED)
    nanobind_add_module(trt_receiver_py src/trt_receiver.cpp src/data_processing.cu)
    target_link_libraries(trt_receiver_py
        nvinfer
        nvonnxparser
        nvparsers
        nvinfer_plugin
    )
endif()
