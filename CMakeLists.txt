cmake_minimum_required(VERSION 3.20)

project(cuda_sfm LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

set(OpenCV_DIR "C:/Dev/opencv/build")
set(EIGEN3_INCLUDE_DIR "C:/Dev/eigen-5.0.1")

find_package(OpenCV REQUIRED)



set(SOURCES
    # CPU
    src/CPU/main.cpp
    src/CPU/CameraIntrinsics.cpp
    src/CPU/FeatureExtractor.cpp
    src/CPU/PointCloudExport.cpp
    src/CPU/PoseRecovery.cpp
    src/CPU/SFM.cpp
    src/CPU/util.cpp

    # Host 侧 CUDA 封装
    src/CPU/FeatureMatch.cu
    src/CPU/Triangulation.cu
    src/CPU/EssentialMatrix.cu

    # GPU kernel / device code
    src/GPU/d_FeatureMatch.cu
    src/GPU/findEssentialMat.cu
    src/GPU/TriangulationCUDA.cu
    src/GPU/GPU_util.cu
)

add_executable(cuda_sfm ${SOURCES})

target_include_directories(cuda_sfm PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(cuda_sfm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

set_target_properties(cuda_sfm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda_sfm PRIVATE
    ${OpenCV_LIBS}
)