#pragma once

// =============================================================================
// shm_layout — 共享内存布局（摄像头端 ↔ 推理端契约）
// =============================================================================
// 与摄像头端定义完全一致，修改时需同步更新两端。
// 双缓冲：write_idx 指示最新完成的帧，读者读 data[write_idx]。

#define SHM_NAME "/spinnaker_img"
#define SHM_WIDTH 640
#define SHM_HEIGHT 640
#define SHM_CHANNELS 1
#define SHM_IMG_SIZE (SHM_WIDTH * SHM_HEIGHT * SHM_CHANNELS)

struct ShmImage {
    int write_idx;                         // 0 或 1，指向最新完成的帧
    int width;
    int height;
    int channels;
    unsigned char data[2][SHM_IMG_SIZE];   // 双缓冲
};
