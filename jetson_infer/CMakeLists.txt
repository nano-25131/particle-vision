cmake_minimum_required(VERSION 3.18)

# =============================================================================
# particle_system — 颗粒目标检测+分割系统 (Jetson Orin)
# =============================================================================
# 依赖: OpenCV, TensorRT, CUDA
# 构建: cmake -B build && cmake --build build
# =============================================================================

project(particle_system LANGUAGES CXX CUDA)

# ---- C++ / CUDA 标准 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)         # Jetson Orin: sm_87

# ---- 编译模式 ----
set(CMAKE_BUILD_TYPE RelWithDebInfo)     # 优化 + 调试符号（Nsight 友好）

# ---- C++ 编译参数 ----
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

# ---- CUDA 编译参数 ----
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")       # Nsight 行号信息
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math") # 快速数学
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- TensorRT / CUDA 路径 (Jetson aarch64) ----
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda/include
    /usr/include/aarch64-linux-gnu
)

link_directories(
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
)

# ---- 可执行文件 ----
add_executable(particle_demo
    # 入口
    src/main.cpp

    # YOLO 检测器（TensorRT + GPU 预处理）
    src/detector/yolo_trt.cu

    # UNet 分割器（TensorRT + GPU 后处理 + 动态 Batch）
    src/segmentor/unet_trt.cu
    src/segmentor/segmentor.cpp

    # 异步流水线
    src/pipeline/pipeline.cpp

    # 共享内存图像源（摄像头）
    src/source/shm_reader.cpp

    # WebSocket 服务（图像推送上位机）
    src/server/ws_server.cpp

    # 工具
    src/utils/image_utils.cpp
)

# ---- 统一 include 路径 ----
target_include_directories(particle_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ---- CUDA 源文件声明 ----
set_source_files_properties(
    src/detector/yolo_trt.cu
    src/segmentor/unet_trt.cu
    PROPERTIES LANGUAGE CUDA
)

# ---- 链接库 ----
target_link_libraries(particle_demo
    ${OpenCV_LIBS}
    nvinfer
    cudart
    rt
    pthread
    ssl
    crypto
)

# ---- Jetson 推荐策略 ----
set(CMAKE_POLICY_DEFAULT_CMP0104 NEW)
