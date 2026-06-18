#include "utils/image_utils.h"
#include <opencv2/opencv.hpp>

namespace image_utils {

// =============================================================================
// pad_to_square — 居中填充为正方形
// =============================================================================
// 将任意尺寸图像放入以 max(w,h) 为边长的正方形中，居中放置，四周填 0。
// 用于 UNet 输入预处理，避免直接 resize 导致长宽比失真。
cv::Mat pad_to_square(const cv::Mat& img, int& x_offset, int& y_offset)
{
    int w = img.cols;
    int h = img.rows;
    int side = std::max(w, h);

    x_offset = (side - w) / 2;
    y_offset = (side - h) / 2;

    cv::Mat square = cv::Mat::zeros(side, side, img.type());
    img.copyTo(square(cv::Rect(x_offset, y_offset, w, h)));
    return square;
}

// =============================================================================
// resize_to — 缩放至指定尺寸
// =============================================================================
cv::Mat resize_to(const cv::Mat& img, int w, int h)
{
    cv::Mat out;
    cv::resize(img, out, cv::Size(w, h));
    return out;
}

// =============================================================================
// to_gray — 转为灰度图
// =============================================================================
// 若已是单通道则直接返回浅拷贝，避免不必要开销
cv::Mat to_gray(const cv::Mat& img)
{
    if (img.channels() == 1)
        return img;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

} // namespace image_utils
