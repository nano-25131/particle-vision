#include "pipeline/pipeline.h"
#include "server/ws_server.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <chrono>
#include <csignal>
#include <cstring>

static volatile sig_atomic_t g_live_running = 1;
static void live_sigint_handler(int) { g_live_running = 0; }

static float read_temp(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) return -1.0f;
    int temp;
    file >> temp;
    return temp / 1000.0f;
}

Pipeline::Pipeline(YoloDetector& yolo, Segmentor& segmentor)
    : yolo_(yolo)
    , segmentor_(segmentor)
{
    stream_thread_ = std::thread(&Pipeline::stream_worker, this);
}

Pipeline::~Pipeline()
{
    stream_queue_.done.store(true);
    stream_queue_.cv.notify_one();
    if (stream_thread_.joinable())
        stream_thread_.join();
}

void Pipeline::process_frame(const cv::Mat& image, int frame_idx)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    // 步骤 1：发射 YOLO（yolo_stream）
    yolo_.detect_async(const_cast<cv::Mat&>(image));

    // 步骤 2：发射 UNet（unet_stream，使用上一帧检测结果）
    bool has_prev = !prev_image_.empty();
    bool has_roi = has_prev && !prev_detections_.empty();
    if (has_roi) {
        std::vector<cv::Rect> roi_boxes;
        roi_boxes.reserve(prev_detections_.size());
        for (auto& det : prev_detections_) {
            cv::Rect valid = det.box & cv::Rect(0, 0, prev_image_.cols, prev_image_.rows);
            if (valid.width > 0 && valid.height > 0)
                roi_boxes.push_back(valid);
        }

        if (!roi_boxes.empty()) {
            mask_full_ = cv::Mat::zeros(prev_image_.size(), CV_8UC1);
            segmentor_.segment_async(prev_image_, roi_boxes, mask_full_);
        } else {
            has_roi = false;
        }
    }

    // 步骤 3：收割 YOLO
    auto detections = yolo_.detect_finish();
    total_detections_ += detections.size();

    // 步骤 4：收割 UNet（如果有），推送可视化
    if (has_prev) {
        if (has_roi) {
            segmentor_.segment_finish();
        } else {
            // 无检测时直接生成空 mask，仍然推图
            mask_full_ = cv::Mat::zeros(prev_image_.size(), CV_8UC1);
        }
        int fps = static_cast<int>(1000.0 / std::max(1.0,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t_start).count() / 1000.0));
        dispatch_visualization(prev_image_, mask_full_, prev_detections_, fps);
    }

    // 步骤 5：交换缓冲区
    prev_image_      = image;
    prev_detections_ = std::move(detections);

    auto t_end = std::chrono::high_resolution_clock::now();
    double frame_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        t_end - t_start).count() / 1000.0;

    total_time_ms_ += frame_ms;
    valid_frames_++;

    std::cout << "[帧 " << frame_idx << "] 检测: " << prev_detections_.size()
              << " 目标 | " << frame_ms << " ms | "
              << (1000.0 / frame_ms) << " FPS" << std::endl;
}

void Pipeline::stream_worker()
{
    while (true) {
        FrameOutput frame;
        {
            std::unique_lock<std::mutex> lock(stream_queue_.mtx);
            stream_queue_.cv.wait(lock, [this] {
                return stream_queue_.has_frame || stream_queue_.done.load();
            });

            if (stream_queue_.done.load() && !stream_queue_.has_frame)
                break;
            if (!stream_queue_.has_frame)
                continue;

            frame = std::move(stream_queue_.latest);
            stream_queue_.has_frame = false;
        }

        // ---- 1. JPEG 编码原图 ----
        std::vector<uchar> jpeg_orig;
        cv::imencode(".jpg", frame.image, jpeg_orig,
                     {cv::IMWRITE_JPEG_QUALITY, 80});

        // ---- 2. 生成 Fused 图 (原图 + 彩色mask叠加 + 检测框) + JPEG 编码 ----
        cv::Mat color_mask;
        cv::applyColorMap(frame.mask, color_mask, cv::COLORMAP_JET);

        cv::Mat fused;
        cv::addWeighted(frame.image, 0.7, color_mask, 0.3, 0, fused);

        for (auto& det : frame.detections)
            cv::rectangle(fused, det.box, cv::Scalar(0, 255, 0), 2);

        std::vector<uchar> jpeg_fused;
        cv::imencode(".jpg", fused, jpeg_fused,
                     {cv::IMWRITE_JPEG_QUALITY, 85});

        // ---- 3. 构建 JSON metadata ----
        auto ts = std::chrono::system_clock::now().time_since_epoch().count();

        std::string json;
        json.reserve(512 + frame.detections.size() * 64);
        json += "{\"fps\":" + std::to_string(frame.fps);
        json += ",\"detections\":" + std::to_string(frame.detections.size());
        json += ",\"timestamp\":" + std::to_string(ts);
        json += ",\"cpu_temp\":" + std::to_string(read_temp("/sys/class/thermal/thermal_zone0/temp"));
        json += ",\"gpu_temp\":" + std::to_string(read_temp("/sys/class/thermal/thermal_zone1/temp"));
        json += ",\"boxes\":[";

        for (size_t i = 0; i < frame.detections.size(); i++) {
            auto& d = frame.detections[i];
            if (i > 0) json += ',';
            json += "{\"x\":" + std::to_string(d.box.x);
            json += ",\"y\":" + std::to_string(d.box.y);
            json += ",\"w\":" + std::to_string(d.box.width);
            json += ",\"h\":" + std::to_string(d.box.height);
            json += ",\"conf\":" + std::to_string(d.conf);
            json += ",\"cls\":" + std::to_string(d.class_id) + "}";
        }
        json += "]}";

        // ---- 4. 构造 binary message ----
        uint32_t json_len  = json.size();
        uint32_t jpeg1_len = jpeg_orig.size();
        uint32_t jpeg2_len = jpeg_fused.size();

        size_t total = 4 + json_len + 4 + jpeg1_len + 4 + jpeg2_len;
        std::vector<uint8_t> msg(total);
        uint8_t* p = msg.data();

        auto write_u32 = [&p](uint32_t v) {
            *p++ = (v >> 24) & 0xFF;
            *p++ = (v >> 16) & 0xFF;
            *p++ = (v >> 8)  & 0xFF;
            *p++ = v & 0xFF;
        };

        write_u32(json_len);
        std::memcpy(p, json.data(), json_len);  p += json_len;
        write_u32(jpeg1_len);
        std::memcpy(p, jpeg_orig.data(), jpeg1_len);  p += jpeg1_len;
        write_u32(jpeg2_len);
        std::memcpy(p, jpeg_fused.data(), jpeg2_len);  p += jpeg2_len;

        // ---- 5. WebSocket 广播 ----
        ws_broadcast(msg.data(), msg.size());
    }
}

void Pipeline::run_live(ShmReader& reader)
{
    if (!reader.is_ready()) {
        std::cerr << "[Pipeline] 共享内存未就绪" << std::endl;
        return;
    }

    std::cout << "[Pipeline] 摄像头连续模式启动 (Ctrl+C 停止)" << std::endl;

    signal(SIGINT, live_sigint_handler);

    int frame_idx = 0;
    while (g_live_running) {
        cv::Mat gray = reader.read();

        if (frame_idx == 0) {
            std::cout << "\n========== 输入图像信息 ==========\n";
            std::cout << "尺寸   : " << gray.cols << " x " << gray.rows << "\n";
            std::cout << "通道数 : " << gray.channels() << "\n";
            std::cout << "类型   : " << gray.type() << " (CV_8UC1=" << CV_8UC1 << ")\n";
            std::cout << "是否连续: " << (gray.isContinuous() ? "是" : "否") << "\n";
            std::cout << "数据指针: " << (void*)gray.data << "\n";
            std::cout << "中心像素值: " << (int)gray.at<unsigned char>(320, 320) << "\n";
            std::cout << "==================================\n\n";
        }

        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);

        process_frame(bgr, frame_idx);
        frame_idx++;
    }
}

void Pipeline::dispatch_visualization(const cv::Mat& image, const cv::Mat& mask,
                                       const std::vector<Detection>& dets, int fps)
{
    FrameOutput fo;
    fo.image      = image.clone();
    fo.mask       = mask.clone();
    fo.detections = dets;
    fo.fps        = fps;

    {
        std::lock_guard<std::mutex> lock(stream_queue_.mtx);
        stream_queue_.latest = std::move(fo);     // 覆盖旧帧，只保留最新
        stream_queue_.has_frame = true;
        stream_queue_.cv.notify_one();
    }
}

double Pipeline::avg_fps() const
{
    if (valid_frames_ == 0) return 0.0;
    return 1000.0 / (total_time_ms_ / valid_frames_);
}

int Pipeline::total_frames() const
{
    return valid_frames_;
}

int Pipeline::total_detections() const
{
    return total_detections_;
}
