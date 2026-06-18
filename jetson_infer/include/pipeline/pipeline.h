#pragma once

#include "common/types.h"
#include "detector/yolo_trt.h"
#include "segmentor/segmentor.h"
#include "source/shm_reader.h"

#include <opencv2/opencv.hpp>

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

// =============================================================================
// Pipeline — 检测+分割异步流水线 (双 GPU 流重叠)
// =============================================================================
//   yolo_stream:  [==== YOLO 帧N ====]              [==== YOLO 帧N+1 ====]
//   unet_stream:    [========= UNet 帧N-1 ==========]

class Pipeline {
public:
    Pipeline(YoloDetector& yolo, Segmentor& segmentor);
    ~Pipeline();

    void run_live(ShmReader& reader);

    double avg_fps()      const;
    int    total_frames() const;
    int    total_detections() const;

private:
    void process_frame(const cv::Mat& image, int frame_idx);

    struct FrameOutput {
        cv::Mat image;
        cv::Mat mask;
        std::vector<Detection> detections;
        int fps = 0;
    };

    struct StreamQueue {
        FrameOutput latest;            // 只保留最新一帧（覆盖旧帧）
        bool has_frame = false;
        std::mutex mtx;
        std::condition_variable cv;
        std::atomic<bool> done{false};
    };

    void stream_worker();
    void dispatch_visualization(const cv::Mat& image, const cv::Mat& mask,
                                const std::vector<Detection>& dets, int fps);

    YoloDetector& yolo_;
    Segmentor&    segmentor_;

    cv::Mat prev_image_;
    std::vector<Detection> prev_detections_;
    cv::Mat mask_full_;

    StreamQueue stream_queue_;
    std::thread    stream_thread_;

    double total_time_ms_ = 0.0;
    int    valid_frames_   = 0;
    int    total_detections_ = 0;
};
