#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>

// =============================================================================
// WebSocketServer — 最小化 WebSocket 服务端（自研，无第三方依赖）
// =============================================================================
//
// 仅实现服务端需要的协议子集:
//   - HTTP Upgrade 握手 (SHA1 via OpenSSL)
//   - 发送 Text/Binary Frame（服务端帧无需 mask）
//   - poll() 事件循环 + 多客户端管理
//
// 线程安全: broadcast() 可从任意线程调用，server_loop 运行在后台线程。

class WebSocketServer {
public:
    WebSocketServer();
    ~WebSocketServer();

    void start(int port);
    void stop();
    void broadcast(const uint8_t* data, size_t len);

    bool is_running() const { return running_.load(); }

private:
    void server_loop();
    bool do_handshake(int fd, const char* request, size_t len);
    bool send_frame(int fd, const uint8_t* data, size_t len, int opcode);
    void disconnect(int fd);

    static std::string compute_accept_key(const std::string& client_key);

    int server_fd_ = -1;
    int port_ = 0;
    std::atomic<bool> running_{false};
    std::thread thread_;

    std::mutex mutex_;
    std::vector<int> client_fds_;
};

// ---- 全局单例 ----
void start_ws_server(int port = 8080);
void stop_ws_server();
void ws_broadcast(const uint8_t* data, size_t len);
