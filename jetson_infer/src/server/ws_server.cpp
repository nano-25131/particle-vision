#include "server/ws_server.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <openssl/sha.h>

// =============================================================================
// Base64 编码（WebSocket 握手 Accept Key 用）
// =============================================================================
static std::string base64_encode(const unsigned char* data, size_t len)
{
    static const char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        unsigned n = (data[i] << 16);
        if (i + 1 < len) n |= (data[i + 1] << 8);
        if (i + 2 < len) n |= data[i + 2];

        out.push_back(chars[(n >> 18) & 0x3F]);
        out.push_back(chars[(n >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? chars[(n >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? chars[n & 0x3F] : '=');
    }
    return out;
}

// =============================================================================
// compute_accept_key — SHA1(client_key + magic GUID) → Base64
// =============================================================================
std::string WebSocketServer::compute_accept_key(const std::string& client_key)
{
    static const char magic[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string combined = client_key + magic;

    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(combined.data()),
         combined.size(), hash);

    return base64_encode(hash, SHA_DIGEST_LENGTH);
}

// =============================================================================
// 全局单例
// =============================================================================
static WebSocketServer* g_ws_server = nullptr;

void start_ws_server(int port)
{
    if (!g_ws_server) {
        g_ws_server = new WebSocketServer();
        g_ws_server->start(port);
    }
}

void stop_ws_server()
{
    if (g_ws_server) {
        g_ws_server->stop();
        delete g_ws_server;
        g_ws_server = nullptr;
    }
}

void ws_broadcast(const uint8_t* data, size_t len)
{
    if (g_ws_server)
        g_ws_server->broadcast(data, len);
}

// =============================================================================
// 构造 / 析构
// =============================================================================
WebSocketServer::WebSocketServer() = default;

WebSocketServer::~WebSocketServer()
{
    stop();
}

// =============================================================================
// start — 创建 socket + bind + listen + 启动后台线程
// =============================================================================
void WebSocketServer::start(int port)
{
    port_ = port;

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "[WS] socket() 失败" << std::endl;
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(server_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[WS] bind() 失败" << std::endl;
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    if (listen(server_fd_, 8) < 0) {
        std::cerr << "[WS] listen() 失败" << std::endl;
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    running_.store(true);
    thread_ = std::thread(&WebSocketServer::server_loop, this);

    std::cout << "[WS] WebSocket 服务启动于 0.0.0.0:" << port_ << std::endl;
}

// =============================================================================
// stop — 停止事件循环 + 关闭所有连接
// =============================================================================
void WebSocketServer::stop()
{
    running_.store(false);

    // shutdown server fd 让 poll() 中的 accept 立即返回
    if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
    }

    if (thread_.joinable())
        thread_.join();

    // 关闭所有客户端连接
    std::lock_guard<std::mutex> lock(mutex_);
    for (int fd : client_fds_)
        close(fd);
    client_fds_.clear();
}

// =============================================================================
// broadcast — 线程安全广播二进制帧给所有客户端
// =============================================================================
void WebSocketServer::broadcast(const uint8_t* data, size_t len)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = client_fds_.begin();
    while (it != client_fds_.end()) {
        if (send_frame(*it, data, len, 0x2)) {  // 0x2 = binary
            ++it;
        } else {
            close(*it);
            it = client_fds_.erase(it);
        }
    }
}

// =============================================================================
// send_frame — 构造 WebSocket 帧并发送
// =============================================================================
// 帧格式（server→client，无 mask）:
//   Byte 0:   FIN(1) | opcode(4)            = 0x80 | opcode
//   Byte 1:   payload_len (7 或 126/127)
//   [2 或 8 bytes: 扩展长度]
//   [payload]
bool WebSocketServer::send_frame(int fd, const uint8_t* payload, size_t len, int opcode)
{
    uint8_t header[10];
    size_t header_size;

    header[0] = 0x80 | (opcode & 0x0F);  // FIN + opcode

    if (len < 126) {
        header[1] = len;
        header_size = 2;
    } else if (len < 65536) {
        header[1] = 126;
        header[2] = (len >> 8) & 0xFF;
        header[3] = len & 0xFF;
        header_size = 4;
    } else {
        header[1] = 127;
        for (int i = 0; i < 8; i++)
            header[2 + i] = (len >> (56 - i * 8)) & 0xFF;
        header_size = 10;
    }

    // 发送帧头
    ssize_t n = ::send(fd, header, header_size, MSG_NOSIGNAL);
    if (n != (ssize_t)header_size)
        return false;

    // 发送 payload
    n = ::send(fd, payload, len, MSG_NOSIGNAL);
    return n == (ssize_t)len;
}

// =============================================================================
// do_handshake — 解析 HTTP Upgrade 请求，返回 101 响应
// =============================================================================
bool WebSocketServer::do_handshake(int fd, const char* request, size_t len)
{
    std::string req(request, len);

    // 提取 Sec-WebSocket-Key
    const char* key_hdr = "Sec-WebSocket-Key: ";
    auto pos = req.find(key_hdr);
    if (pos == std::string::npos) return false;

    pos += strlen(key_hdr);
    auto end = req.find("\r\n", pos);
    if (end == std::string::npos) return false;

    std::string key = req.substr(pos, end - pos);
    std::string accept = compute_accept_key(key);

    char response[256];
    int n = snprintf(response, sizeof(response),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n"
        "\r\n",
        accept.c_str());

    return ::send(fd, response, n, MSG_NOSIGNAL) == n;
}

// =============================================================================
// disconnect — 从客户端列表中移除并关闭 fd
// =============================================================================
void WebSocketServer::disconnect(int fd)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(client_fds_.begin(), client_fds_.end(), fd);
    if (it != client_fds_.end())
        client_fds_.erase(it);
    close(fd);
}

// =============================================================================
// server_loop — poll() 事件循环（后台线程）
// =============================================================================
void WebSocketServer::server_loop()
{
    while (running_.load()) {
        // 构建 pollfd 数组
        std::vector<pollfd> fds;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            fds.reserve(1 + client_fds_.size());

            pollfd pfd;
            pfd.fd = server_fd_;
            pfd.events = POLLIN;
            fds.push_back(pfd);

            for (int fd : client_fds_) {
                pfd.fd = fd;
                pfd.events = POLLIN;
                fds.push_back(pfd);
            }
        }

        int ret = poll(fds.data(), fds.size(), 100);  // 100ms 超时，可响应 stop
        if (ret < 0) continue;
        if (ret == 0) continue;  // 超时，重新检查 running_

        for (size_t i = 0; i < fds.size(); i++) {
            if (fds[i].revents == 0) continue;

            if (fds[i].fd == server_fd_) {
                // ---- 新连接 ----
                if (fds[i].revents & POLLIN) {
                    int client = accept(server_fd_, nullptr, nullptr);
                    if (client < 0) continue;

                    // 设置 TCP_NODELAY 减少小包延迟
                    int opt = 1;
                    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

                    // 读握手请求
                    char buf[4096];
                    ssize_t n = recv(client, buf, sizeof(buf) - 1, 0);
                    if (n > 0) {
                        buf[n] = '\0';
                        if (do_handshake(client, buf, n)) {
                            std::lock_guard<std::mutex> lock(mutex_);
                            client_fds_.push_back(client);
                            std::cout << "[WS] 客户端连接 fd=" << client << std::endl;
                            continue;
                        }
                    }
                    close(client);
                }
            } else {
                // ---- 已有客户端 ----
                if (fds[i].revents & (POLLHUP | POLLERR | POLLNVAL)) {
                    disconnect(fds[i].fd);
                } else if (fds[i].revents & POLLIN) {
                    // 客户端发了数据（可能是 close frame 或 ping）
                    // 读一下确认连接状态，不做业务处理
                    char buf[256];
                    ssize_t n = recv(fds[i].fd, buf, sizeof(buf), 0);
                    if (n <= 0)
                        disconnect(fds[i].fd);
                }
            }
        }
    }
}
