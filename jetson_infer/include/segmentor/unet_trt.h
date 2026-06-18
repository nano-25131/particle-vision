#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

// =============================================================================
// UnetTRT — UNet 语义分割推理器（TensorRT + GPU 后处理 + 动态 Batch）
// =============================================================================
//
// 功能:
//   - 批量推理：一次处理 N 张 128×128 灰度 ROI，输出二值 mask
//   - GPU 端 argmax 后处理（float→u8 mask），避免 GPU→CPU 大体积传输
//   - 页锁定内存 + 异步传输，最小化 H2D/D2H 延迟
//   - 动态 Batch 支持（max_batch 由引擎优化配置文件决定）
//
// 异步接口使用模式:
//   unet.infer_async(roi_list);       // 发射 GPU 工作（unet_stream），立即返回
//   // ... 在此期间 YOLO 可以并发执行 ...
//   auto masks = unet.infer_finish();  // 同步 + 返回 mask 列表
//
// 同步便捷接口:
//   auto masks = unet.infer_batch(roi_list);  // = async + finish
//
// 数据流（GPU 端全程）:
//   CPU u8 ROI → pinned H2D → gpu_u8_staging_
//     → u8_to_float_kernel → gpu_float_input_ (TRT)
//     → TRT enqueueV3 → gpu_float_output_
//     → argmax_2class_kernel → gpu_u8_mask_
//     → D2H pinned → CPU cv::Mat
class UnetTRT {
public:
    // engine_path: TensorRT .engine 文件路径
    // max_batch:   请求的最大批量（会被引擎优化配置文件的实际上限裁剪）
    explicit UnetTRT(const std::string& engine_path, int max_batch = 8);
    ~UnetTRT();

    // ---- 异步管线接口 ----
    // 发射 CPU 预处理 + H2D + TRT 推理 + argmax + D2H，全部异步在 unet_stream 上
    void infer_async(const std::vector<cv::Mat>& roi_list);

    // 同步 unet_stream，包装结果为 cv::Mat 列表
    std::vector<cv::Mat> infer_finish();

    // ---- 同步便捷接口 ----
    std::vector<cv::Mat> infer_batch(const std::vector<cv::Mat>& roi_list);

    // ---- 状态查询 ----
    int max_batch() const { return max_batch_; }
    cudaStream_t get_stream() const { return stream_; }

private:
    void parse_io_shapes();    // 从引擎解析 I/O 维度
    void allocate_buffers();   // 按 max_batch_ 分配所有 GPU/CPU 缓冲区

    // ---- TensorRT 对象 ----
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;           // UNet 专用 CUDA 流（与 YOLO 流独立）

    // 张量名称
    std::string input_name_;
    std::string output_name_;

    // ---- GPU 缓冲区 ----
    void* gpu_float_input_  = nullptr;        // TRT 输入 (float32, NCHW)
    void* gpu_float_output_ = nullptr;        // TRT 输出 (float32, NCHW, 2 类)
    unsigned char* gpu_u8_staging_ = nullptr; // u8 输入暂存（H2D 后、float 转换前）
    unsigned char* gpu_u8_mask_    = nullptr; // argmax 输出 (u8 mask, NHW)

    // ---- 页锁定 CPU 缓冲区（DMA 直接访问，加速 H2D/D2H） ----
    unsigned char* pinned_input_  = nullptr;  // 输入 (u8, NCHW)
    unsigned char* pinned_output_ = nullptr;  // 输出 (u8 mask, NHW)

    // ---- 输入/输出尺寸 ----
    int in_c_ = 0, in_h_ = 0, in_w_ = 0;      // 输入: 1×128×128
    int out_c_ = 0, out_h_ = 0, out_w_ = 0;    // 输出: 2×128×128

    // ---- 动态 Batch ----
    int max_batch_ = 0;             // 引擎优化配置文件实际上限
    size_t single_input_elems_  = 0;  // 单张输入元素数 (C×H×W)
    size_t single_output_elems_ = 0;  // 单张输出元素数 (C×H×W)

    // ---- 异步 pending 状态 ----
    int pending_batch_ = 0;         // infer_async 存储当前批大小，infer_finish 使用
};
