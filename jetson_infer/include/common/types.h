#pragma once
#include <opencv2/opencv.hpp>

// =============================================================================
// Detection — 单目标检测结果
// =============================================================================
// 由 YOLO 检测器输出，同时作为 UNet 分割的 ROI 来源。
// 在流水线中流转路径为:
//   YOLO.detect_finish() → Pipeline.prev_detections_ → Segmentor.segment_async()
struct Detection {
    cv::Rect box;   // 边界框（原图坐标）
    float conf;     // 置信度 [0, 1]
    int class_id;   // 类别 ID（0: 颗粒, 1: ...）
};
