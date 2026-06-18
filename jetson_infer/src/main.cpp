#include "detector/yolo_trt.h"
#include "segmentor/segmentor.h"
#include "pipeline/pipeline.h"
#include "source/shm_reader.h"
#include "server/ws_server.h"

#include <iostream>

int main()
{
    // ———— 加载模型 ————
    std::cout << "加载 YOLO 检测模型..." << std::endl;
    YoloDetector yolo("../model/yolo_new_fp16.engine");
    if (!yolo.is_ready()) {
        std::cerr << "YOLO 初始化失败！" << std::endl;
        return -1;
    }

    std::cout << "加载 UNet 分割模型..." << std::endl;
    Segmentor segmentor("../model/unet.engine");

    // ———— 共享内存 ————
    ShmReader reader;
    if (!reader.is_ready()) {
        std::cerr << "摄像头共享内存未就绪，请先启动摄像头端程序" << std::endl;
        return -1;
    }

    // ———— WebSocket 服务 ————
    start_ws_server(8080);

    // ———— 运行流水线 ————
    Pipeline pipeline(yolo, segmentor);
    pipeline.run_live(reader);

    // ———— 清理 ————
    stop_ws_server();

    // ———— 汇总 ————
    std::cout << "\n================================\n";
    std::cout << "平均 FPS : " << pipeline.avg_fps() << "\n";
    std::cout << "处理帧数 : " << pipeline.total_frames() << "\n";
    std::cout << "总检测数 : " << pipeline.total_detections() << "\n";
    std::cout << "平均目标 : " << (pipeline.total_detections()
                                 / std::max(1, pipeline.total_frames()))
              << " 个/帧\n";
    std::cout << "================================\n";

    return 0;
}
