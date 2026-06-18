#pragma once

#include "common/types.h"

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

// =============================================================================
// YoloDetector — YOLO 目标检测器（TensorRT + GPU 预处理 + 异步管线）
// =============================================================================
//
// 功能:
//   - 加载 TensorRT 序列化引擎，执行 YOLO 目标检测
//   - GPU 端 BGR→RGB 颜色转换 + 归一化（融合为单一 CUDA kernel）
//   - 异步管线接口，支持与下游 UNet 流并发执行
//
// 异步接口使用模式:
//   yolo.detect_async(img);          // 发射 GPU 工作（yolo_stream），立即返回
//   // ... 在此期间可以发射其他 GPU 工作 ...
//   auto dets = yolo.detect_finish(); // 同步 + 后处理，返回检测结果
//
// 同步便捷接口:
//   auto dets = yolo.detect(img);     // = async + finish
//
// 内部 CUDA 资源:
//   stream_       — YOLO 专用 CUDA 流，与 UNet 流独立
//   buffers_[2]   — TensorRT 输入/输出缓冲（float32 planar CHW）
//   gpu_u8_input_  — HWC u8 中间缓冲（cv::resize 输出）
//   pinned_resized_ — 页锁定内存（加速 H2D 传输）
class YoloDetector {
public:
    // ---- 构造 / 析构 ----
    // engine_path: TensorRT .engine 文件路径
    explicit YoloDetector(const std::string& engine_path);
    ~YoloDetector();

    // ---- 异步管线接口 ----
    // 发射 GPU 预处理 + TRT 推理 + D2H，全部异步在 yolo_stream 上
    void detect_async(cv::Mat& img);

    // 同步 yolo_stream，执行 CPU 后处理，返回检测结果
    std::vector<Detection> detect_finish();

    // ---- 同步便捷接口 ----
    std::vector<Detection> detect(cv::Mat& img);

    // ---- 状态查询 ----
    bool is_ready() const;                        // 引擎是否加载成功
    cudaStream_t get_stream() const { return stream_; }

private:
    // ---- 初始化步骤 ----
    bool load_engine(const std::string& path);     // 反序列化 TensorRT 引擎
    bool create_context();                         // 创建执行上下文
    bool setup_tensor_info();                      // 解析 I/O 维度、分配大小
    bool allocate_buffers();                       // 分配 GPU/CPU 缓冲区

    // ---- 处理 ----
    void preprocess_gpu(cv::Mat& img, float& scale);   // GPU 预处理（resize + BGR→RGB + 归一化）
    std::vector<Detection> postprocess(float* output, float scale,
                                       int img_w, int img_h);  // CPU NMS 后处理
    void print_engine_info();                              // 日志输出

    // ---- TensorRT 对象 ----
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // 张量名称（从引擎自动检测，支持不同导出命名）
    std::string input_name_;
    std::string output_name_;

    // ---- CUDA 资源 ----
    void* buffers_[2] = {nullptr, nullptr};   // [0]: TRT 输入, [1]: TRT 输出
    unsigned char* gpu_u8_input_ = nullptr;    // 中间 u8 缓冲（HWC，用于 H2D 前暂存）
    unsigned char* pinned_resized_ = nullptr;  // 页锁定内存（DMA 加速 H2D）
    std::vector<float> cpu_output_;            // D2H 后的输出缓冲（CPU 端）
    cudaStream_t stream_ = nullptr;            // YOLO 专用 CUDA 流

    // ---- 输出维度 ----
    int num_boxes_   = 0;    // 检测框总数（如 25200 for 640×640）
    int stride_      = 0;    // 每框信息长度 (cx, cy, w, h, obj, cls...)
    int num_classes_ = 0;    // 类别数

    // 缓冲区大小
    size_t input_bytes_  = 0;
    size_t output_bytes_ = 0;

    // ---- 异步 pending 状态 ----
    // detect_async 存储当前帧参数，detect_finish 时使用
    float pending_scale_ = 1.0f;
    int pending_img_w_ = 0;
    int pending_img_h_ = 0;
};
