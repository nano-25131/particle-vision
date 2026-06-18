#include "segmentor/unet_trt.h"
#include "utils/image_utils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

using namespace nvinfer1;

// =============================================================================
// Logger — TensorRT 日志回调（匿名命名空间，避免与 yolo_trt.cu 的 ODR 冲突）
// =============================================================================
namespace {

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[UNet-TRT] " << msg << std::endl;
    }
} g_logger;

} // anonymous namespace

// =============================================================================
// GPU 核函数
// =============================================================================

// u8 → float32，归一化到 [0, 1]
// 输入: 灰度 u8 图像 (每个元素 0-255)
// 输出: float32 (每个元素 0.0-1.0)
static __global__ void u8_to_float_kernel(
    const unsigned char* src, float* dst, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        dst[idx] = src[idx] * (1.0f / 255.0f);
}

// 逐像素 argmax（2 类）: 通道1 > 通道0 → 255 (前景)，否则 0 (背景)
// UNet 输出 2 通道 logits，取 argmax 得到二值 mask
static __global__ void argmax_2class_kernel(
    const float* src, unsigned char* dst,
    int hw, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hw;
    if (idx >= total) return;

    int b = idx / hw;       // 批次内索引
    int p = idx % hw;       // 像素位置

    // 通道 0 (背景) 与通道 1 (前景) 的 logit 值
    float c0 = src[b * 2 * hw + p];       // 通道 0
    float c1 = src[b * 2 * hw + hw + p];  // 通道 1
    dst[idx] = (c1 > c0) ? 255 : 0;
}

// =============================================================================
// 构造 — 加载引擎 → 创建上下文 → 解析维度 → 分配缓冲
// =============================================================================
UnetTRT::UnetTRT(const std::string& engine_path, int max_batch)
{
    max_batch_ = max_batch;

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("[UNet] 无法打开引擎文件: " + engine_path);

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> data(size);
    file.read(data.data(), size);

    runtime_ = createInferRuntime(g_logger);
    engine_  = runtime_->deserializeCudaEngine(data.data(), size);
    context_ = engine_->createExecutionContext();

    // 自动检测输入/输出张量名称
    for (int i = 0; i < engine_->getNbIOTensors(); i++) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT)
            input_name_ = name;
        else
            output_name_ = name;
    }

    cudaStreamCreate(&stream_);

    parse_io_shapes();   // 解析引擎实际维度，裁剪 max_batch_
    allocate_buffers();  // 按裁剪后的 max_batch_ 分配

    std::cout << "[UNet] 初始化成功！(GPU后处理) 最大Batch=" << max_batch_ << std::endl;
}

// =============================================================================
// parse_io_shapes — 从引擎解析 I/O 张量维度
// =============================================================================
void UnetTRT::parse_io_shapes()
{
    auto in_dims  = engine_->getTensorShape(input_name_.c_str());
    auto out_dims = engine_->getTensorShape(output_name_.c_str());

    // UNet 输入: [batch, 1, 128, 128]
    // UNet 输出: [batch, 2, 128, 128]
    in_c_ = in_dims.d[1];  in_h_ = in_dims.d[2];  in_w_ = in_dims.d[3];
    out_c_ = out_dims.d[1]; out_h_ = out_dims.d[2]; out_w_ = out_dims.d[3];

    std::cout << "\n========== UNet 输入输出尺寸 ==========\n";
    std::cout << "输入: " << in_c_ << "x" << in_h_ << "x" << in_w_ << "\n";
    std::cout << "输出: " << out_c_ << "x" << out_h_ << "x" << out_w_ << "\n";
    std::cout << "最大 Batch: " << max_batch_ << "\n";
    std::cout << "========================================\n\n";

    single_input_elems_  = in_c_ * in_h_ * in_w_;
    single_output_elems_ = out_c_ * out_h_ * out_w_;
}

// =============================================================================
// allocate_buffers — 按 max_batch_ 一次性分配所有 GPU/CPU 缓冲区
// =============================================================================
void UnetTRT::allocate_buffers()
{
    int mb = max_batch_;

    // GPU 缓冲区
    cudaMalloc(&gpu_float_input_,  mb * single_input_elems_  * sizeof(float));
    cudaMalloc(&gpu_float_output_, mb * single_output_elems_ * sizeof(float));
    cudaMalloc(&gpu_u8_staging_,   mb * single_input_elems_  * sizeof(unsigned char));
    cudaMalloc(&gpu_u8_mask_,      mb * out_h_ * out_w_ * sizeof(unsigned char));

    // 页锁定 CPU 缓冲区（DMA 直接访问，加速 H2D/D2H）
    cudaHostAlloc(&pinned_input_,  mb * single_input_elems_ * sizeof(unsigned char),
                  cudaHostAllocDefault);
    cudaHostAlloc(&pinned_output_, mb * out_h_ * out_w_ * sizeof(unsigned char),
                  cudaHostAllocDefault);

    // 预绑定 TensorRT I/O 地址（固定不变，无需每帧重新绑定）
    context_->setTensorAddress(input_name_.c_str(),  gpu_float_input_);
    context_->setTensorAddress(output_name_.c_str(), gpu_float_output_);
}

// =============================================================================
// infer_async — 异步发射 GPU 工作（全部在 unet_stream 上排队）
// =============================================================================
// 流程:
//   CPU: resize→grayscale→pinned memcpy
//   GPU: H2D → u8→float → TRT enqueueV3 → argmax → D2H (全部异步)
//
// 注意: 不能并发调用两次 infer_async（内部 pending_ 状态会被覆盖）
void UnetTRT::infer_async(const std::vector<cv::Mat>& roi_list)
{
    int batch = roi_list.size();
    if (batch <= 0) return;
    if (batch > max_batch_)
        throw std::runtime_error("批量大小 " + std::to_string(batch) +
                                 " 超过 max_batch " + std::to_string(max_batch_));

    pending_batch_ = batch;

    // ---- 步骤 1：CPU 预处理 → 直接写入页锁定内存 ----
    // 页锁定内存可被 DMA 直接访问，H2D 速度快于普通可分页内存
    size_t total_u8_elems = batch * single_input_elems_;
    for (int b = 0; b < batch; b++) {
        cv::Mat resized = image_utils::resize_to(roi_list[b], in_w_, in_h_);
        resized = image_utils::to_gray(resized);
        std::memcpy(pinned_input_ + b * single_input_elems_,
                    resized.data, single_input_elems_);
    }

    // ---- 步骤 2：H2D (页锁定 → GPU u8 staging，异步 DMA) ----
    cudaMemcpyAsync(gpu_u8_staging_, pinned_input_, total_u8_elems,
                    cudaMemcpyHostToDevice, stream_);

    // ---- 步骤 3：GPU kernel: u8 → float32 归一化 ----
    int block = 256;
    int grid  = (total_u8_elems + block - 1) / block;
    u8_to_float_kernel<<<grid, block, 0, stream_>>>(
        gpu_u8_staging_, static_cast<float*>(gpu_float_input_), total_u8_elems);

    // ---- 步骤 4：设置动态 Batch + TRT 推理 ----
    Dims4 input_shape(batch, in_c_, in_h_, in_w_);
    context_->setInputShape(input_name_.c_str(), input_shape);
    context_->enqueueV3(stream_);

    // ---- 步骤 5：GPU kernel: float logits → u8 mask (argmax) ----
    int hw = out_h_ * out_w_;
    int total_pixels = batch * hw;
    grid = (total_pixels + block - 1) / block;
    argmax_2class_kernel<<<grid, block, 0, stream_>>>(
        static_cast<float*>(gpu_float_output_), gpu_u8_mask_, hw, batch);

    // ---- 步骤 6：D2H (GPU u8 mask → 页锁定内存，异步 DMA) ----
    cudaMemcpyAsync(pinned_output_, gpu_u8_mask_, total_pixels,
                    cudaMemcpyDeviceToHost, stream_);

    // 不在此处同步！infer_finish() 负责收割
}

// =============================================================================
// infer_finish — 同步收割（等待 unet_stream + 包装结果）
// =============================================================================
std::vector<cv::Mat> UnetTRT::infer_finish()
{
    // 等待 unet_stream 上所有操作完成
    cudaStreamSynchronize(stream_);

    // 从页锁定内存包装为 cv::Mat（clone 避免引用 pinned 内存）
    int hw = out_h_ * out_w_;
    std::vector<cv::Mat> masks;
    masks.reserve(pending_batch_);
    for (int b = 0; b < pending_batch_; b++) {
        cv::Mat mask(out_h_, out_w_, CV_8UC1, pinned_output_ + b * hw);
        masks.push_back(mask.clone());
    }

    return masks;
}

// =============================================================================
// infer_batch — 同步便捷接口
// =============================================================================
std::vector<cv::Mat> UnetTRT::infer_batch(const std::vector<cv::Mat>& roi_list)
{
    infer_async(roi_list);
    return infer_finish();
}

// =============================================================================
// 析构 — 按依赖顺序逆序释放所有 CUDA 资源
// =============================================================================
UnetTRT::~UnetTRT()
{
    // 先释放 GPU 内存
    if (gpu_float_input_)  cudaFree(gpu_float_input_);
    if (gpu_float_output_) cudaFree(gpu_float_output_);
    if (gpu_u8_staging_)   cudaFree(gpu_u8_staging_);
    if (gpu_u8_mask_)      cudaFree(gpu_u8_mask_);
    // 再释放页锁定 CPU 内存
    if (pinned_input_)     cudaFreeHost(pinned_input_);
    if (pinned_output_)    cudaFreeHost(pinned_output_);
    // 最后销毁流
    cudaStreamDestroy(stream_);

    delete context_;
    delete engine_;
    delete runtime_;
}
