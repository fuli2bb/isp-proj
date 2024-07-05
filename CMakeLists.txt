cmake_minimum_required(VERSION 3.10)
project(isp-proj LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(OpenCV_DIR "/home/fuli2/opencv_build/opencv/build")
set(CUDA_TOOLKIT_ROOT_DIR "/software/cuda-11.6")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Automatically include all .cpp and .cu files from the include directory
file(GLOB INCLUDE_CPP_FILES "include/*.cpp")
file(GLOB INCLUDE_CUDA_FILES "include/*.cu")

# Specify include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} include)

# Create a list of CUDA source files
set(CUDA_SOURCES ${INCLUDE_CUDA_FILES})

# Specify the source files for the project
set(SOURCES src/isp.cpp ${INCLUDE_CPP_FILES})

# Set NVCC flags for multiple architectures
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75")

# Add the CUDA source files to the project and compile with NVCC
add_library(ISP_CUDA STATIC ${CUDA_SOURCES})
set_target_properties(ISP_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Create the executable
add_executable(ISP ${SOURCES})

# Link the OpenCV and CUDA libraries
target_link_libraries(ISP PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ISP_CUDA)

