#pragma once
#include <opencv2/opencv.hpp>

// =============================================================================
// image_utils — 图像预处理工具函数
// =============================================================================
// 为 UNet 推理提供 ROI 裁剪→填充正方形→缩放→灰度化的标准化流程。
// 所有函数均为纯 CPU 操作，不涉及 GPU。
namespace image_utils {

    // 居中填充为正方形，不拉伸原图
    // x_offset, y_offset: 返回原图在正方形中的左上角偏移，用于反向映射
    cv::Mat pad_to_square(const cv::Mat& img, int& x_offset, int& y_offset);

    // 缩放至指定尺寸（双线性插值）
    cv::Mat resize_to(const cv::Mat& img, int w, int h);

    // 转为灰度图（若已是单通道则直接返回，避免不必要拷贝）
    cv::Mat to_gray(const cv::Mat& img);

} // namespace image_utils
