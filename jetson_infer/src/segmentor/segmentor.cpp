#include "segmentor/segmentor.h"
#include <iostream>
#include <algorithm>

// =============================================================================
// 构造 — 初始化 UNet 推理器
// =============================================================================
Segmentor::Segmentor(const std::string& unet_engine_path)
    : unet_(unet_engine_path)
{
    std::cout << "[Segmentor] 初始化完成，UNet max_batch=" << unet_.max_batch() << std::endl;
}

// =============================================================================
// segment_async — 异步发射（只发射首个 chunk 的 GPU 工作）
// =============================================================================
//
// 策略:
//   - 一次性 CPU 预处理所有 ROI（resize→pad→gray），存储到 pending_
//   - 首个 chunk（≤max_batch 个 ROI）发射 unet.infer_async，GPU 工作异步执行
//   - 后续 chunk 留给 segment_finish 同步处理
//
// 为什么只发射首个 chunk:
//   - pinned_input_ 是单缓冲，同一时刻只能有一个 chunk 在 GPU 上
//   - 首个 chunk 异步后立即返回，让 GPU 开始工作，后面的 CPU 工作可并行
void Segmentor::segment_async(const cv::Mat& image,
                              const std::vector<cv::Rect>& roi_boxes,
                              cv::Mat& mask_full)
{
    if (roi_boxes.empty()) return;

    int total = roi_boxes.size();
    int max_batch = unet_.max_batch();

    // 保存上下文，供 segment_finish 继续使用
    pending_.image     = &image;
    pending_.mask_full = &mask_full;
    pending_.roi_boxes = roi_boxes;
    pending_.total     = total;

    // ---- CPU 预处理所有 ROI（一次性做完，结果缓存到 pending_） ----
    pending_.square_batch.clear();
    pending_.square_batch.reserve(total);
    pending_.valid_rects.clear();
    pending_.valid_rects.reserve(total);

    std::vector<cv::Mat> roi_batch;    // 首个 chunk 的 UNet 输入
    roi_batch.reserve(max_batch);

    for (int i = 0; i < total; i++) {
        const cv::Rect& box = roi_boxes[i];
        cv::Mat roi = image(box);   // 浅拷贝（共享原图数据）

        // 填充为正方形（避免 resize 失真）
        int x_off = 0, y_off = 0;
        cv::Mat square = image_utils::pad_to_square(roi, x_off, y_off);

        // 缩放到 UNet 输入尺寸 128×128 → 灰度
        cv::Mat input = image_utils::resize_to(square, 128, 128);
        input = image_utils::to_gray(input);

        pending_.square_batch.push_back(square);
        pending_.valid_rects.push_back(cv::Rect(x_off, y_off, roi.cols, roi.rows));

        // 只填充首个 chunk
        if (i < max_batch)
            roi_batch.push_back(input);
    }

    // ---- 发射首个 chunk 的 GPU 工作（异步，立即返回） ----
    unet_.infer_async(roi_batch);
}

// =============================================================================
// segment_finish — 收割 + 处理剩余 chunk + 融合
// =============================================================================
//
// 步骤:
//   1. 收割首个 chunk（unet.infer_finish），融合到 mask_full
//   2. 处理剩余 chunk（每批完整 CPU→GPU→CPU 同步周期），逐个融合
//
// 反向映射链:
//   UNet mask (128×128) → resize → 正方形 → 裁剪 ROI 区域 → resize → 原图 bbox
//   融合方式: cv::max(dst, mask_bbox, dst)，多个 ROI 重叠区域取最大值
void Segmentor::segment_finish()
{
    if (pending_.total <= 0) return;

    int max_batch = unet_.max_batch();
    const cv::Mat& image     = *pending_.image;
    cv::Mat& mask_full       = *pending_.mask_full;
    auto& roi_boxes          = pending_.roi_boxes;
    auto& square_batch       = pending_.square_batch;
    auto& valid_rects        = pending_.valid_rects;

    int chunk_start = 0;
    while (chunk_start < pending_.total) {
        int chunk_size = std::min(max_batch, pending_.total - chunk_start);

        std::vector<cv::Mat> masks;

        if (chunk_start == 0) {
            // 首个 chunk: segment_async 已发射 GPU 工作，只需收割
            masks = unet_.infer_finish();
        } else {
            // 后续 chunk: 完整预处理 + 同步推理
            std::vector<cv::Mat> roi_batch;
            roi_batch.reserve(chunk_size);

            for (int j = 0; j < chunk_size; j++) {
                int idx = chunk_start + j;
                const cv::Rect& box = roi_boxes[idx];
                cv::Mat roi = image(box);

                int x_off = 0, y_off = 0;
                cv::Mat square = image_utils::pad_to_square(roi, x_off, y_off);

                cv::Mat input = image_utils::resize_to(square, 128, 128);
                input = image_utils::to_gray(input);

                roi_batch.push_back(input);
                // 更新缓存（保守策略：重新计算以确保一致性）
                square_batch[idx] = square;
                valid_rects[idx] = cv::Rect(x_off, y_off, roi.cols, roi.rows);
            }

            masks = unet_.infer_batch(roi_batch);
        }

        // ---- 反向映射 + 融合 ----
        for (int j = 0; j < chunk_size; j++) {
            int idx = chunk_start + j;

            // 1. mask → 正方形尺寸
            cv::Mat mask_square;
            cv::resize(masks[j], mask_square, square_batch[idx].size(),
                       0, 0, cv::INTER_NEAREST);

            // 2. 裁剪回 ROI 区域
            cv::Mat mask_roi = mask_square(valid_rects[idx]).clone();

            // 3. 缩放至原图 bbox 尺寸
            cv::Mat mask_bbox;
            cv::resize(mask_roi, mask_bbox, roi_boxes[idx].size(),
                       0, 0, cv::INTER_NEAREST);

            // 4. 融合到全图 mask（cv::max: 多个 ROI 重叠区域取 255）
            cv::Mat dst = mask_full(roi_boxes[idx]);
            cv::max(dst, mask_bbox, dst);
        }

        chunk_start += chunk_size;
    }

    // 清空 pending 状态
    pending_.total = 0;
}

// =============================================================================
// segment_batch — 同步便捷接口
// =============================================================================
void Segmentor::segment_batch(const cv::Mat& image,
                              const std::vector<cv::Rect>& roi_boxes,
                              cv::Mat& mask_full)
{
    segment_async(image, roi_boxes, mask_full);
    segment_finish();
}
