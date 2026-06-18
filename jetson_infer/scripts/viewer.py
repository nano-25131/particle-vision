#!/usr/bin/env python3
"""
上位机图像查看器 — WebSocket 接收 Jetson 推理结果
用法:
    python3 viewer.py <jetson-ip>
    python3 viewer.py 192.168.1.100
    python3 viewer.py 192.168.1.100:8080

依赖: pip3 install websocket-client opencv-python numpy
"""

import sys
import json
import struct
import time
import threading
import cv2
import numpy as np
import websocket


def main():
    if len(sys.argv) < 2:
        print("用法: python3 viewer.py <jetson-ip>")
        print("示例: python3 viewer.py 192.168.1.100")
        sys.exit(1)

    addr = sys.argv[1]
    if ":" in addr and not addr.startswith("ws"):
        addr = f"ws://{addr}/"
    elif not addr.startswith("ws"):
        addr = f"ws://{addr}:8080/"

    print(f"连接 {addr} ...")

    state = {
        "orig": None,
        "mask": None,
        "fps": 0,
        "detections": 0,
        "boxes": [],
        "cpu_temp": -1,
        "gpu_temp": -1,
        "display_fps": 0.0,
        "connected": False,
    }
    lock = threading.Lock()

    # 类别颜色表 (BGR) — 按需扩展
    CLASS_COLORS = [
        (0, 255, 0),    # 0: 绿色
        (255, 0, 0),    # 1: 蓝色
        (0, 0, 255),    # 2: 红色
        (255, 255, 0),  # 3: 青色
        (255, 0, 255),  # 4: 品红
        (0, 255, 255),  # 5: 黄色
        (128, 0, 255),  # 6: 橙
        (255, 128, 0),  # 7: 天蓝
    ]

    cv2.namedWindow("original", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def on_message(ws, message):
        if not isinstance(message, bytes):
            return

        try:
            # 解析 binary format
            data = message
            off = 0

            json_len = struct.unpack('>I', data[off:off+4])[0]
            off += 4

            meta = json.loads(data[off:off+json_len])
            off += json_len

            jpeg1_len = struct.unpack('>I', data[off:off+4])[0]
            off += 4

            jpeg1 = data[off:off+jpeg1_len]
            off += jpeg1_len

            jpeg2_len = struct.unpack('>I', data[off:off+4])[0]
            off += 4

            jpeg2 = data[off:off+jpeg2_len]

            # 解码图像
            orig = cv2.imdecode(np.frombuffer(jpeg1, dtype=np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(np.frombuffer(jpeg2, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            with lock:
                state["orig"] = orig
                state["mask"] = mask
                state["fps"] = meta.get("fps", 0)
                state["detections"] = meta.get("detections", 0)
                state["boxes"] = meta.get("boxes", [])
                state["cpu_temp"] = meta.get("cpu_temp", -1)
                state["gpu_temp"] = meta.get("gpu_temp", -1)
                state["connected"] = True

        except Exception as e:
            print(f"解析错误: {e}")

    def on_open(ws):
        print("已连接")

    def on_close(ws, code, msg):
        print(f"连接断开 (code={code})")
        with lock:
            state["connected"] = False

    def on_error(ws, err):
        print(f"错误: {err}")

    ws = websocket.WebSocketApp(
        addr,
        on_message=on_message,
        on_open=on_open,
        on_close=on_close,
        on_error=on_error,
    )

    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()

    t_last = time.time()

    while True:
        with lock:
            orig = state["orig"]
            mask = state["mask"]
            fps = state["fps"]
            detections = state["detections"]
            boxes = state["boxes"][:]
            cpu_temp = state["cpu_temp"]
            gpu_temp = state["gpu_temp"]
            connected = state["connected"]
            display_fps = state["display_fps"]

        # 渲染原图 + 检测框
        if orig is not None:
            orig_disp = orig.copy()
            for b in boxes:
                cls = b.get("cls", 0)
                color = CLASS_COLORS[cls % len(CLASS_COLORS)]

                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cv2.rectangle(orig_disp, (x, y), (x+w, y+h), color, 2)
                label = f"cls{cls} {b['conf']:.2f}"
                cv2.putText(orig_disp, label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            info = (
                f"FPS: {fps} | Display: {display_fps:.0f} | "
                f"Detections: {detections} | "
                f"CPU: {cpu_temp:.1f}C | GPU: {gpu_temp:.1f}C"
            )
            if not connected:
                info += " | DISCONNECTED"
            cv2.putText(orig_disp, info, (10, orig_disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("original", orig_disp)

        # 渲染 mask
        if mask is not None:
            cv2.imshow("mask", mask)

        key = cv2.waitKey(16) & 0xFF
        if key == ord('q') or key == 27:
            break

        t_now = time.time()
        dfps = 1.0 / max(t_now - t_last, 0.001)
        with lock:
            state["display_fps"] = state["display_fps"] * 0.8 + dfps * 0.2
        t_last = t_now

    ws.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
