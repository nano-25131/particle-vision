#include "source/shm_reader.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <iostream>

ShmReader::ShmReader()
{
    shm_fd_ = shm_open(SHM_NAME, O_RDONLY, 0);
    if (shm_fd_ < 0) {
        std::cerr << "[ShmReader] 无法打开共享内存 " << SHM_NAME
                  << " (摄像头端是否已启动?)" << std::endl;
        return;
    }

    shm_ = static_cast<ShmImage*>(
        mmap(nullptr, sizeof(ShmImage), PROT_READ, MAP_SHARED, shm_fd_, 0));

    if (shm_ == MAP_FAILED) {
        std::cerr << "[ShmReader] mmap 失败" << std::endl;
        shm_ = nullptr;
        return;
    }

    std::cout << "[ShmReader] 共享内存已映射, "
              << shm_->width << "x" << shm_->height
              << " ch=" << shm_->channels << std::endl;
}

ShmReader::~ShmReader()
{
    if (shm_ && shm_ != MAP_FAILED)
        munmap(shm_, sizeof(ShmImage));
    if (shm_fd_ >= 0)
        close(shm_fd_);
}

bool ShmReader::is_ready() const
{
    return shm_ != nullptr && shm_ != MAP_FAILED;
}

cv::Mat ShmReader::read()
{
    int idx = shm_->write_idx;
    // 直接包装 shm data 为 cv::Mat（零拷贝）
    return cv::Mat(shm_->height, shm_->width, CV_8UC1, shm_->data[idx]);
}
