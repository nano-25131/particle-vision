#include "Spinnaker.h"
  #include "SpinGenApi/SpinnakerGenApi.h"
  #include <opencv2/opencv.hpp>
  #include <iostream>
  #include <sys/mman.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <cstring>
  #include <algorithm>

  using namespace Spinnaker;
  using namespace Spinnaker::GenApi;
  using namespace std;

  #define SHM_NAME "/spinnaker_img"
  #define SHM_WIDTH 640
  #define SHM_HEIGHT 640
  #define SHM_CHANNELS 1
  #define SHM_IMG_SIZE (SHM_WIDTH * SHM_HEIGHT * SHM_CHANNELS)

  // 🔥 双缓冲
  struct ShmImage {
      int write_idx;
      int width;
      int height;
      int channels;
      unsigned char data[2][SHM_IMG_SIZE];
  };

  // ================= 采集函数 =================
  int AcquireImages(CameraPtr pCam, ShmImage* shm)
  {
      int result = 0;

      try
      {
          ImageProcessor processor;
          processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

          cout << "Start acquiring images..." << endl;

          while (true)
          {
              try
              {
                  ImagePtr pResultImage = pCam->GetNextImage(50);

                  if (!pResultImage->IsIncomplete())
                  {
                      ImagePtr converted = processor.Convert(pResultImage, PixelFormat_Mono8);

                      // Convert 是深拷贝，之后不再需要 pResultImage，立即归还 buffer
                      pResultImage->Release();

                      cv::Mat cvImage(
                          (int)converted->GetHeight(),
                          (int)converted->GetWidth(),
                          CV_8UC1,
                          converted->GetData()
                      );

                      // 等比例缩放：保持宽高比，适配到 640×640 内
                      int cam_w = cvImage.cols;
                      int cam_h = cvImage.rows;
                      double scale = std::min(640.0 / cam_w, 640.0 / cam_h);
                      int new_w = (int)(cam_w * scale);
                      int new_h = (int)(cam_h * scale);

                      cv::Mat resized;
                      cv::resize(cvImage, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

                      // 居中填充黑边到 640×640
                      int top    = (640 - new_h) / 2;
                      int bottom = 640 - new_h - top;
                      int left   = (640 - new_w) / 2;
                      int right  = 640 - new_w - left;

                      cv::Mat netImage;
                      cv::copyMakeBorder(resized, netImage, top, bottom, left, right,
                                         cv::BORDER_CONSTANT, cv::Scalar(0));

                      CV_Assert(netImage.isContinuous());

                      // ========== 双缓冲写 ==========
                      int next = 1 - shm->write_idx;

                      memcpy(
                          shm->data[next],
                          netImage.data,
                          SHM_IMG_SIZE
                      );

                      // 🔥 最后切换（关键）
                      shm->write_idx = next;
                  }
                  else
                  {
                      // 不完整的帧也要释放
                      pResultImage->Release();
                  }
              }
              catch (Spinnaker::Exception &e)
              {
                  cout << "Image error: " << e.what() << endl;
              }
          }

          pCam->EndAcquisition();
      }
      catch (Spinnaker::Exception &e)
      {
          cout << "Error: " << e.what() << endl;
          result = -1;
      }

      return result;
  }

  // ================= 主函数 =================
  int main()
  {
      SystemPtr system = System::GetInstance();
      CameraList camList = system->GetCameras();

      if (camList.GetSize() == 0)
      {
          cout << "No camera found!" << endl;
          return -1;
      }

      // ===== 共享内存 =====
      int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
      ftruncate(shm_fd, sizeof(ShmImage));

      ShmImage* shm = (ShmImage*)mmap(
          nullptr,
          sizeof(ShmImage),
          PROT_READ | PROT_WRITE,
          MAP_SHARED,
          shm_fd,
          0
      );

      if (shm == MAP_FAILED)
      {
          perror("mmap");
          return -1;
      }

      shm->width = SHM_WIDTH;
      shm->height = SHM_HEIGHT;
      shm->channels = SHM_CHANNELS;
      shm->write_idx = 0;

      // ===== 相机 =====
      CameraPtr pCam = camList.GetByIndex(0);
      pCam->Init();

      INodeMap &nodeMap = pCam->GetNodeMap();

      // 🔥 在 BeginAcquisition 之前设置
      CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
      if (IsWritable(ptrAcquisitionMode))
      {
          ptrAcquisitionMode->SetIntValue(
              ptrAcquisitionMode->GetEntryByName("Continuous")->GetValue()
          );
          cout << "Acquisition mode set to Continuous\n";
      }
      else
      {
          cout << "Warning: AcquisitionMode not writable\n";
      }

      // 🔥 开始采集（一定在设置之后）
      pCam->BeginAcquisition();

      // ===== 开始循环 =====
      AcquireImages(pCam, shm);

      // 清理
      pCam->EndAcquisition();
      pCam->DeInit();

      munmap(shm, sizeof(ShmImage));
      close(shm_fd);

      camList.Clear();
      system->ReleaseInstance();

      return 0;
  }