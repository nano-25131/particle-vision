#pragma once

#include "common/shm_layout.h"

#include <opencv2/opencv.hpp>

// =============================================================================
// ShmReader — 共享内存图像读取器
// =============================================================================
// 从 /spinnaker_img 共享内存读取摄像头帧（Mono8 640×640）。
// cv::Mat 直接引用 shm data，零拷贝，调用方如需持有需 clone。

class ShmReader {
public:
    ShmReader();
    ~ShmReader();

    bool is_ready() const;

    // 读取最新帧（引用 shm data，不 clone）
    cv::Mat read();

private:
    int shm_fd_ = -1;
    ShmImage* shm_ = nullptr;
};
