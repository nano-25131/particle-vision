#include "detector/yolo_trt.h"

#include <fstream>
#include <iostream>
#include <cstring>

using namespace nvinfer1;

// =============================================================================
// Logger — TensorRT 日志回调（匿名命名空间，避免 ODR 冲突）
// =============================================================================
namespace {

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 仅输出 WARNING 及以上级别
        if (severity <= Severity::kWARNING)
            std::cout << "[YOLO-TRT] " << msg << std::endl;
    }
} g_logger;

} // anonymous namespace

// =============================================================================
// GPU 核函数: BGR u8 HWC → RGB float32 planar CHW，归一化到 [0, 1]
// =============================================================================
// 一个 kernel 完成三件事，避免多次内存访问:
//   1. 通道重排 (BGR → RGB)
//   2. uint8 → float32
//   3. 归一化 (÷255)
// 输入:  u8[3×H×W] HWC 格式（OpenCV cv::Mat 默认布局）
// 输出:  float[3×H×W] planar CHW 格式（TensorRT 期望布局）
static __global__ void bgr_to_rgb_planar_kernel(
    const unsigned char* src, float* dst, int area)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= area) return;

    int src_idx = idx * 3;
    // OpenCV 默认 BGR 顺序: src[0]=B, src[1]=G, src[2]=R
    dst[0 * area + idx] = src[src_idx + 2] * (1.0f / 255.0f);  // R
    dst[1 * area + idx] = src[src_idx + 1] * (1.0f / 255.0f);  // G
    dst[2 * area + idx] = src[src_idx + 0] * (1.0f / 255.0f);  // B
}

// =============================================================================
// 构造 — 加载引擎 → 创建上下文 → 解析维度 → 分配缓冲
// =============================================================================
YoloDetector::YoloDetector(const std::string& engine_path)
{
    if (!load_engine(engine_path)) return;
    if (!create_context())    return;
    if (!setup_tensor_info()) return;
    if (!allocate_buffers())  return;

    // CPU 输出缓冲：一次性分配，detect_finish 中复用
    cpu_output_.resize(output_bytes_ / sizeof(float));

    std::cout << "[YOLO] 初始化成功！(GPU 预处理, 异步管线)" << std::endl;
}

// =============================================================================
// 析构 — 按依赖顺序逆序释放所有 CUDA 资源
// =============================================================================
YoloDetector::~YoloDetector()
{
    // 先销毁流（等待流中所有操作完成）
    if (stream_)        cudaStreamDestroy(stream_);
    // 再释放 GPU 内存
    if (buffers_[0])     cudaFree(buffers_[0]);
    if (buffers_[1])     cudaFree(buffers_[1]);
    if (gpu_u8_input_)   cudaFree(gpu_u8_input_);
    // 最后释放页锁定 CPU 内存
    if (pinned_resized_) cudaFreeHost(pinned_resized_);

    delete context_;
    delete engine_;
    delete runtime_;
}

// =============================================================================
// load_engine — 从文件反序列化 TensorRT 引擎
// =============================================================================
bool YoloDetector::load_engine(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[YOLO] 引擎文件未找到: " << path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> model(size);
    file.read(model.data(), size);
    file.close();

    runtime_ = createInferRuntime(g_logger);
    engine_  = runtime_->deserializeCudaEngine(model.data(), size);
    if (!engine_) {
        std::cerr << "[YOLO] 引擎反序列化失败！" << std::endl;
        return false;
    }
    return true;
}

// =============================================================================
// create_context — 创建推理执行上下文
// =============================================================================
bool YoloDetector::create_context()
{
    context_ = engine_->createExecutionContext();
    return context_ != nullptr;
}

// =============================================================================
// setup_tensor_info — 解析 I/O 张量名称、设置输入形状、计算缓冲大小
// =============================================================================
bool YoloDetector::setup_tensor_info()
{
    // 自动检测输入/输出张量名称（兼容不同导出命名）
    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; i++) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT)
            input_name_ = name;
        else
            output_name_ = name;
    }

    // 设置固定输入形状 (batch=1, 3ch, 640×640)
    context_->setInputShape(input_name_.c_str(), Dims4(1, 3, 640, 640));

    // 调用 inferShapes 让 TensorRT 推导输出维度
    context_->inferShapes(0, nullptr);

    auto out_dims = context_->getTensorShape(output_name_.c_str());
    // YOLO 输出: [1, num_boxes, stride]，其中 stride = 5 + num_classes
    num_boxes_   = out_dims.d[1];
    stride_      = out_dims.d[2];
    num_classes_ = stride_ - 5;  // cx, cy, w, h, obj_conf, cls...

    int out_count = 1;
    for (int i = 0; i < out_dims.nbDims; i++)
        out_count *= out_dims.d[i];

    input_bytes_  = 3 * 640 * 640 * sizeof(float);
    output_bytes_ = out_count * sizeof(float);

    print_engine_info();
    return true;
}

// =============================================================================
// allocate_buffers — 一次性分配所有 GPU/CPU 缓冲区 + 创建 CUDA 流
// =============================================================================
bool YoloDetector::allocate_buffers()
{
    // GPU 推理缓冲
    cudaMalloc(&buffers_[0], input_bytes_);
    cudaMalloc(&buffers_[1], output_bytes_);

    // 中间 GPU 缓冲（u8 HWC，用于接收 cv::resize 的 H2D）
    cudaMalloc(&gpu_u8_input_, 640 * 640 * 3 * sizeof(unsigned char));

    // 页锁定 CPU 内存（DMA 加速 H2D）
    cudaHostAlloc(&pinned_resized_, 640 * 640 * 3 * sizeof(unsigned char),
                  cudaHostAllocDefault);

    // 创建 YOLO 专用 CUDA 流
    cudaStreamCreate(&stream_);

    // 预绑定 TensorRT 张量地址（固定不变，无需每帧重新绑定）
    context_->setTensorAddress(input_name_.c_str(), buffers_[0]);
    context_->setTensorAddress(output_name_.c_str(), buffers_[1]);

    return true;
}

// =============================================================================
// preprocess_gpu — GPU 端预处理
// =============================================================================
// 流程: CPU resize → 页锁定内存 → H2D (async) → GPU kernel (BGR→RGB+归一化)
// 输入 img: 原图（BGR u8 HWC）
// 输出 scale: 缩放因子（用于后处理反向映射检测框）
void YoloDetector::preprocess_gpu(cv::Mat& img, float& scale)
{
    // 等比例缩放: 长边对齐 640
    scale = std::min(640.0f / img.cols, 640.0f / img.rows);

    // CPU resize 直接写入页锁定内存（DMA 可直接访问，加速后续 H2D）
    cv::Mat resized(640, 640, CV_8UC3, pinned_resized_);
    cv::resize(img, resized, cv::Size(640, 640));

    // 异步 H2D
    cudaMemcpyAsync(gpu_u8_input_, pinned_resized_, 640 * 640 * 3,
                    cudaMemcpyHostToDevice, stream_);

    // GPU kernel: BGR u8 HWC → RGB float32 planar CHW → 直接写入 TRT 输入缓冲
    int area  = 640 * 640;
    int block = 256;
    int grid  = (area + block - 1) / block;
    bgr_to_rgb_planar_kernel<<<grid, block, 0, stream_>>>(
        gpu_u8_input_, static_cast<float*>(buffers_[0]), area);
}

// =============================================================================
// detect_async — 异步发射（GPU 工作全部在 yolo_stream 上排队）
// =============================================================================
// 调用后立即返回，GPU 工作异步执行。
// 必须在 detect_finish 之前完成。不能并发调用两次 detect_async
//（内部 pending_ 状态会被覆盖）。
void YoloDetector::detect_async(cv::Mat& img)
{
    // 预处理（核函数发射 + H2D 都在 stream_ 上排队）
    preprocess_gpu(img, pending_scale_);
    pending_img_w_ = img.cols;
    pending_img_h_ = img.rows;

    // TRT 推理（张量地址已在构造时绑定）
    context_->enqueueV3(stream_);

    // D2H 下载（异步，完成后 cpu_output_ 可在 detect_finish 中读取）
    cudaMemcpyAsync(cpu_output_.data(), buffers_[1], output_bytes_,
                    cudaMemcpyDeviceToHost, stream_);
}

// =============================================================================
// detect_finish — 同步收割（等待 yolo_stream + CPU 后处理）
// =============================================================================
std::vector<Detection> YoloDetector::detect_finish()
{
    // 等待 yolo_stream 上所有操作完成
    cudaStreamSynchronize(stream_);

    return postprocess(cpu_output_.data(), pending_scale_,
                       pending_img_w_, pending_img_h_);
}

// =============================================================================
// detect — 同步便捷接口
// =============================================================================
std::vector<Detection> YoloDetector::detect(cv::Mat& img)
{
    detect_async(img);
    return detect_finish();
}

// =============================================================================
// postprocess — CPU 端后处理（置信度过滤 + NMS）
// =============================================================================
std::vector<Detection> YoloDetector::postprocess(float* output, float scale,
                                                  int img_w, int img_h)
{
    constexpr float kConfThresh = 0.25f;  // 目标置信度阈值
    constexpr float kNmsThresh  = 0.45f;  // NMS IoU 阈值

    std::vector<Detection> results;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    boxes.reserve(128);
    scores.reserve(128);
    class_ids.reserve(128);

    // 遍历所有检测框
    for (int i = 0; i < num_boxes_; i++) {
        float* row = output + i * stride_;

        // 目标置信度初筛
        float obj_conf = row[4];
        if (obj_conf < kConfThresh) continue;

        // 找最佳类别
        int best_cls = 0;
        float best_cls_conf = 0.f;
        for (int c = 0; c < num_classes_; c++) {
            if (row[5 + c] > best_cls_conf) {
                best_cls_conf = row[5 + c];
                best_cls = c;
            }
        }

        // 综合置信度 = obj_conf × cls_conf
        float conf = obj_conf * best_cls_conf;
        if (conf < kConfThresh) continue;

        // 中心点格式 → 左上角格式，同时反向缩放
        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        boxes.push_back(cv::Rect(
            (int)((cx - w * 0.5f) / scale),
            (int)((cy - h * 0.5f) / scale),
            (int)(w / scale),
            (int)(h / scale)
        ));
        scores.push_back(conf);
        class_ids.push_back(best_cls);
    }

    // NMS 去重
    std::vector<int> nms_idx;
    cv::dnn::NMSBoxes(boxes, scores, kConfThresh, kNmsThresh, nms_idx);

    results.reserve(nms_idx.size());
    for (int idx : nms_idx)
        results.push_back({boxes[idx], scores[idx], class_ids[idx]});

    return results;
}

// =============================================================================
// print_engine_info — 引擎信息日志
// =============================================================================
void YoloDetector::print_engine_info()
{
    std::cout << "\n========== YOLO 引擎信息 ==========\n";
    std::cout << "输入  : " << input_name_  << "\n";
    std::cout << "输出  : " << output_name_ << "\n";
    std::cout << "框数  : " << num_boxes_   << "\n";
    std::cout << "步长  : " << stride_      << "\n";
    std::cout << "类别数: " << num_classes_ << "\n";
    std::cout << "====================================\n\n";
}

// =============================================================================
// is_ready — 查询引擎是否可用
// =============================================================================
bool YoloDetector::is_ready() const
{
    return engine_ && context_;
}
