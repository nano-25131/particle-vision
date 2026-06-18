# Particle — 颗粒目标实时检测与分割系统

实时粒子推理系统：Blackfly 工业相机采集 → Jetson Orin 推理（YOLO + UNet）→ PC 上位机可视化。

## 架构

```
┌──────────────────────┐    共享内存     ┌──────────────────────┐    WebSocket    ┌──────────────────────┐
│  Blackfly USB3 相机   │ ────────────→  │  Jetson Orin (推理端)  │ ────────────→  │   PC 上位机 (显示端)   │
│  jetson_capture/      │  /spinnaker_img │  jetson_infer/        │  binary frames │  pc_viewer/           │
│                       │                 │  YOLO检测 + UNet分割   │  :8080         │  ImGui + ImPlot       │
└──────────────────────┘                 └──────────────────────┘                 └──────────────────────┘
```

## 子项目

| 目录 | 功能 | 平台 | 关键依赖 |
|---|---|---|---|
| `jetson_capture/` | 相机采集，写入共享内存 | Linux (Jetson) | Spinnaker SDK, OpenCV |
| `jetson_infer/` | YOLO + UNet 推理，WebSocket 推送 | Linux (Jetson Orin) | TensorRT, CUDA, OpenCV |
| `pc_viewer/` | 上位机仪表盘，实时可视化 | Linux (PC) | GLFW, ImGui, ImPlot, OpenCV |

## 快速开始

### 1. 相机采集端

```bash
cd jetson_capture
# 安装 Spinnaker SDK 后
cmake -B build && cmake --build build
sudo ./build/bin/SpinnakerAcquisition
```

### 2. Jetson 推理端

```bash
cd jetson_infer
# 准备模型：将 ONNX 导出为 TensorRT engine，放入 model/
#   model/yolo_new_fp16.engine
#   model/unet.engine
cmake -B build && cmake --build build
./build/particle_demo
```

### 3. PC 上位机

```bash
cd pc_viewer
cmake -B build && cmake --build build
./build/viewer <jetson-ip>
# 例如: ./build/viewer 192.168.1.100
```

也可用 Python 脚本快速查看：

```bash
python3 jetson_infer/scripts/viewer.py <jetson-ip>
```

## 数据流

```
相机 640×640 Mono8
  → 共享内存 (/spinnaker_img, 双缓冲)
  → ShmReader 零拷贝读取
  → YOLO 检测 (TensorRT, GPU)
  → UNet 分割 (TensorRT, 动态 Batch, GPU)
  → JPEG 编码 + JSON 元数据
  → WebSocket Binary Frame 广播 :8080
  → pc_viewer ImGui 仪表盘实时渲染
```

## 模型准备

1. 将训练好的 YOLO 模型导出为 ONNX
2. 将训练好的 UNet 模型导出为动态 batch ONNX（batch 维度设为 -1）
3. 使用 `trtexec` 在 Jetson Orin 上转换为 TensorRT engine：

```bash
# YOLO (固定 batch=1)
trtexec --onnx=yolo.onnx --saveEngine=model/yolo_new_fp16.engine --fp16

# UNet (动态 batch, 推荐 max=8)
trtexec --onnx=unet.onnx --saveEngine=model/unet.engine --fp16 \
        --minShapes=input:1x1x128x128 --optShapes=input:4x1x128x128 --maxShapes=input:8x1x128x128
```
