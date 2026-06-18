#pragma once

#include "segmentor/unet_trt.h"
#include "utils/image_utils.h"

#include <opencv2/opencv.hpp>

#include <vector>

// =============================================================================
// Segmentor — 批量分割管理器
// =============================================================================
//
// 功能:
//   - 接收 YOLO 检测结果（ROI 列表），对每个 ROI 执行 UNet 分割
//   - 自动分 chunk（每批不超过 UNet max_batch），批量推理
//   - ROI 预处理流水线: 裁剪 → 填充正方形 → 缩放 128×128 → 灰度
//   - 反向映射: UNet mask → 正方形 → ROI → 全图融合（cv::max）
//
// 异步接口使用模式:
//   segmentor.segment_async(image, roi_boxes, mask_full);
//       // CPU 预处理全部 ROI + 发射首个 chunk 的 UNet GPU 工作，立即返回
//   // ... 在此期间 YOLO 可以并发执行 ...
//   segmentor.segment_finish();
//       // 收割首个 chunk + 处理剩余 chunk + 融合到 mask_full
//
// 同步便捷接口:
//   segmentor.segment_batch(image, roi_boxes, mask_full);  // = async + finish
//
// 注意:
//   - segment_async 必须与 segment_finish 配对调用
//   - segment_async 传入的 image/mask_full 引用必须在 segment_finish
//     完成前保持有效（由 Pipeline 保证）
class Segmentor {
public:
    explicit Segmentor(const std::string& unet_engine_path);

    // ---- 异步管线接口 ----
    // 预处理全部 ROI + 发射首个 chunk 的 UNet 异步推理
    void segment_async(const cv::Mat& image,
                       const std::vector<cv::Rect>& roi_boxes,
                       cv::Mat& mask_full);

    // 收割首个 chunk + 同步处理剩余 chunk + 融合到 mask_full
    void segment_finish();

    // ---- 同步便捷接口 ----
    void segment_batch(const cv::Mat& image,
                       const std::vector<cv::Rect>& roi_boxes,
                       cv::Mat& mask_full);

private:
    UnetTRT unet_;

    // ---- 异步 pending 状态 ----
    // 保存 segment_async 的上下文，供 segment_finish 继续使用
    struct PendingState {
        const cv::Mat* image = nullptr;       // 原始图像（指针，不持有所有权）
        cv::Mat* mask_full = nullptr;         // 分割结果全图（指针，不持有所有权）

        std::vector<cv::Rect> roi_boxes;      // 所有 ROI 框
        std::vector<cv::Mat> square_batch;    // 填充正方形（每个 ROI）
        std::vector<cv::Rect> valid_rects;    // 正方形内的 ROI 偏移

        int total = 0;                        // ROI 总数
    } pending_;
};
