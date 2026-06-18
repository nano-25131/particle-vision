#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <random>

#include <opencv2/opencv.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <openssl/sha.h>

#include "json.hpp"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================================================
// base64 encode (WebSocket 握手 Accept Key)
// ==========================================================================
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

// ==========================================================================
// ring buffer  (固定容量环形缓冲区)
// ==========================================================================
// 想象一个首尾相连的圆圈，上面有 Capacity 个格子。head 指针沿着圆圈
// 顺时针走动，每次写入后往前移动一格。当缓冲区写满后，新的数据会覆盖
// 最旧的数据 —— 这就是"环形"的含义。
//
// 示意图 (Capacity = 5, count = 5, 即已满):
//
//   写入顺序:  A  B  C  D  E  (A最早, E最新)
//   数组布局: [D][E][ ][A][B]
//                    ↑
//                   head  (下一个要写入的位置, 也是"最旧数据"的位置)
//
//   copy_chronological 输出: A, B, C, D, E  (按时间从旧到新)
//
template <int Capacity>
struct RingBuffer {
    float buf[Capacity] = {};   // 固定大小的栈上数组, 编译期确定, 无动态分配
    int   head  = 0;            // 下一次 push 将要写入的索引
    int   count = 0;            // 当前已存储的有效元素个数 (0 .. Capacity)

    // 写入一个新值, 覆盖最旧的数据 (如果已满)
    void push(float v) {
        buf[head] = v;                       // 在 head 处写入
        head = (head + 1) % Capacity;        // head 前进一格, 到头后绕回 0
        if (count < Capacity) count++;        // 未满时计数递增, 满了就保持 Capacity
    }

    // 把缓冲区内容按时间顺序 (从旧到新) 复制到 dst
    // dst 必须至少有 count 个 float 的空间
    void copy_chronological(float* dst) const {
        if (count < Capacity) {
            // 还没绕圈: buf[0] 就是最旧, buf[count-1] 就是最新
            for (int i = 0; i < count; ++i) dst[i] = buf[i];
        } else {
            // 已经绕圈: head 指向的元素就是最旧的那个
            // (因为它马上就要被下一次 push 覆盖)
            int tail = head;
            for (int i = 0; i < count; ++i) {
                dst[i] = buf[tail];
                tail = (tail + 1) % Capacity;
            }
        }
    }
};

// ==========================================================================
// shared state
// ==========================================================================
struct BoxInfo {
    int   x, y, w, h;
    float conf;
    int   cls;
};

struct AppState {
    cv::Mat input;
    cv::Mat fused;
    int    fps         = 0;
    float  fps_smooth  = 0.0f;
    int    detections  = 0;
    float  cpu_temp    = -1.0f;
    float  gpu_temp    = -1.0f;
    std::vector<BoxInfo> boxes;

    std::string host  = "192.168.1.100:8080";
    bool connected    = false;
    bool connecting   = false;

    std::mutex mtx;
};

// ==========================================================================
// OpenGL texture
// ==========================================================================
struct GLTexture {
    GLuint id    = 0;
    int    width = 0, height = 0;

    void upload(const cv::Mat& img) {
        if (img.empty()) return;
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        cv::flip(rgb, rgb, 0);

        if (id == 0) glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        width  = rgb.cols;
        height = rgb.rows;
    }

    void destroy() { if (id) { glDeleteTextures(1, &id); id = 0; } }
    ImTextureID get() const {
        return id ? reinterpret_cast<ImTextureID>(static_cast<intptr_t>(id)) : nullptr;
    }
    float aspect() const {
        return (height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    }
};

// ==========================================================================
// dashboard compute state (updated only when output frame changes)
// ==========================================================================
struct DashboardCompute {
    std::vector<float> hist_bins;   // 32-bin grayscale histogram (normalised 0..1)
    float hist_mean    = 0;
    float hist_stddev  = 0;
    float motion_pct   = 0;         // frame-diff percentage
    cv::Mat diff_thumb;             // small thumbnail of absdiff (16x12)

    cv::Mat prev_output;
    bool    dirty = true;

    void update(const cv::Mat& output) {
        if (output.empty()) { dirty = true; return; }
        cv::Mat gray;
        if (output.channels() == 1)
            gray = output;
        else
            cv::cvtColor(output, gray, cv::COLOR_BGR2GRAY);

        // histogram
        int bins = 32;
        float range[] = {0, 256};
        const float* ranges[] = {range};
        cv::Mat hist_mat;
        cv::calcHist(&gray, 1, nullptr, cv::Mat(), hist_mat, 1, &bins, ranges);
        hist_bins.resize(bins);
        float max_val = 0;
        for (int i = 0; i < bins; ++i) {
            hist_bins[i] = hist_mat.at<float>(i);
            if (hist_bins[i] > max_val) max_val = hist_bins[i];
        }
        if (max_val > 0) for (auto& v : hist_bins) v /= max_val;

        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        hist_mean   = static_cast<float>(mean[0]);
        hist_stddev = static_cast<float>(stddev[0]);

        // frame diff
        if (!prev_output.empty() && prev_output.size() == output.size()) {
            cv::Mat diff;
            cv::absdiff(output, prev_output, diff);
            cv::Scalar diff_mean = cv::mean(diff);
            motion_pct = static_cast<float>(diff_mean[0]) / 255.0f * 100.0f;

            cv::Mat diff_gray;
            cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
            cv::resize(diff_gray, diff_thumb, cv::Size(64, 36), 0, 0, cv::INTER_AREA);
        }
        prev_output = output.clone();
        dirty = false;
    }
};

// ==========================================================================
// minimal WebSocket client
// ==========================================================================
static std::string ws_accept_key(const std::string& client_key)
{
    static const char magic[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string combined = client_key + magic;
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(combined.data()), combined.size(), hash);
    return base64_encode(hash, SHA_DIGEST_LENGTH);
}

static bool ws_handshake(int fd, const std::string& host)
{
    // 生成随机 16 字节 key → base64
    unsigned char key_bytes[16];
    std::random_device rd;
    for (int i = 0; i < 16; ++i) key_bytes[i] = rd() & 0xFF;
    std::string key = base64_encode(key_bytes, 16);

    char req[512];
    snprintf(req, sizeof(req),
        "GET / HTTP/1.1\r\n"
        "Host: %s\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: %s\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n",
        host.c_str(), key.c_str());

    if (::send(fd, req, strlen(req), MSG_NOSIGNAL) != (ssize_t)strlen(req))
        return false;

    char buf[2048];
    ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return false;
    buf[n] = '\0';

    // 检查 101 和 Sec-WebSocket-Accept
    std::string resp(buf, n);
    if (resp.find("101") == std::string::npos) return false;

    auto pos = resp.find("Sec-WebSocket-Accept: ");
    if (pos == std::string::npos) return false;
    pos += 22;
    auto end = resp.find("\r\n", pos);
    std::string accept = resp.substr(pos, end - pos);

    return accept == ws_accept_key(key);
}

// 读取一个 WebSocket 帧，返回 binary payload
// 返回 true 表示成功，false 表示断开/错误
static bool ws_read_frame(int fd, std::vector<uint8_t>& payload)
{
    payload.clear();

    // 读前 2 字节
    uint8_t hdr[2];
    ssize_t n = recv(fd, hdr, 2, MSG_WAITALL);
    if (n != 2) return false;

    uint8_t opcode = hdr[0] & 0x0F;

    // 读扩展长度
    uint64_t len = hdr[1] & 0x7F;
    if (len == 126) {
        uint8_t ext[2];
        if (recv(fd, ext, 2, MSG_WAITALL) != 2) return false;
        len = (ext[0] << 8) | ext[1];
    } else if (len == 127) {
        uint8_t ext[8];
        if (recv(fd, ext, 8, MSG_WAITALL) != 8) return false;
        len = 0;
        for (int i = 0; i < 8; ++i) len = (len << 8) | ext[i];
    }

    // 服务端帧无 mask

    // 安全上限：单帧最大 10MB，防止异常长度导致 bad_alloc
    constexpr uint64_t kMaxFrameSize = 10 * 1024 * 1024;
    if (len > kMaxFrameSize) {
        std::cerr << "[WS] 帧过大 " << len << " 字节，断开" << std::endl;
        return false;
    }

    // 读 payload
    if (len > 0) {
        payload.resize(len);
        size_t off = 0;
        while (off < len) {
            n = recv(fd, payload.data() + off, len - off, 0);
            if (n <= 0) return false;
            off += n;
        }
    }

    // 处理控制帧
    if (opcode == 0x8) {  // close
        // 发送 close 响应
        uint8_t close_frame[] = {0x88, 0x00};
        ::send(fd, close_frame, 2, MSG_NOSIGNAL);
        return false;
    }
    if (opcode == 0x9) {  // ping → pong
        uint8_t pong[2] = {0x8A, (uint8_t)(len > 127 ? 0 : len)};
        ::send(fd, pong, 2, MSG_NOSIGNAL);
        if (len > 0 && len <= 125)
            ::send(fd, payload.data(), len, MSG_NOSIGNAL);
        return ws_read_frame(fd, payload);  // 递归读下一帧
    }
    if (opcode == 0xA)  // pong (ignore)
        return ws_read_frame(fd, payload);

    return opcode == 0x2;  // 只接受 binary 帧
}

// ==========================================================================
// fetch thread (WebSocket)
// ==========================================================================
static void fetch_thread(AppState& state, std::atomic<bool>& stop)
{
    while (!stop.load(std::memory_order_relaxed)) {
        // 解析 host:port
        std::string host_str;
        { std::lock_guard<std::mutex> lk(state.mtx); host_str = state.host; }
        state.connecting = true;

        std::string addr = host_str;
        std::string port = "8080";
        auto colon = addr.find(':');
        if (colon != std::string::npos) {
            port = addr.substr(colon + 1);
            addr = addr.substr(0, colon);
        }

        // DNS 解析
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(addr.c_str(), port.c_str(), &hints, &res) != 0) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        // TCP 连接
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
            freeaddrinfo(res);
            if (fd >= 0) close(fd);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        freeaddrinfo(res);

        // WebSocket 握手
        if (!ws_handshake(fd, host_str)) {
            close(fd);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(state.mtx);
            state.connected  = true;
            state.connecting = false;
        }

        // 读帧循环
        while (!stop.load(std::memory_order_relaxed)) {
            std::vector<uint8_t> payload;
            if (!ws_read_frame(fd, payload)) break;

            try {
                // 解析 binary format: [json_len BE][JSON][jpeg1_len BE][JPEG1][jpeg2_len BE][JPEG2]
                const uint8_t* p = payload.data();
                size_t rem = payload.size();

                auto read_u32 = [&]() -> uint32_t {
                    uint32_t v = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3];
                    p += 4; rem -= 4;
                    return v;
                };
                if (rem < 4) continue;
                uint32_t json_len = read_u32();
                if (rem < json_len) continue;
                std::string json_str(reinterpret_cast<const char*>(p), json_len);
                p += json_len; rem -= json_len;

                auto j = nlohmann::json::parse(json_str);
                int   fps        = j.value("fps", 0);
                int   detections = j.value("detections", 0);
                float cpu_temp   = j.value("cpu_temp", -1.0f);
                float gpu_temp   = j.value("gpu_temp", -1.0f);
                float fps_smooth = 0.0f;
                { std::lock_guard<std::mutex> lk(state.mtx); fps_smooth = state.fps_smooth * 0.7f + fps * 0.3f; }

                std::vector<BoxInfo> boxes;
                for (auto& bj : j.value("boxes", nlohmann::json::array())) {
                    boxes.push_back({
                        bj.value("x", 0), bj.value("y", 0),
                        bj.value("w", 0), bj.value("h", 0),
                        bj.value("conf", 0.0f), bj.value("cls", 0)
                    });
                }

                // JPEG1 (原图)
                cv::Mat input_img;
                if (rem >= 4) {
                    uint32_t jpeg1_len = read_u32();
                    if (rem >= jpeg1_len) {
                        input_img = cv::imdecode(
                            std::vector<uchar>(p, p + jpeg1_len), cv::IMREAD_COLOR);
                        p += jpeg1_len; rem -= jpeg1_len;
                    }
                }

                // JPEG2 (fused)
                cv::Mat fused_img;
                if (rem >= 4) {
                    uint32_t jpeg2_len = read_u32();
                    if (rem >= jpeg2_len) {
                        fused_img = cv::imdecode(
                            std::vector<uchar>(p, p + jpeg2_len), cv::IMREAD_COLOR);
                    }
                }

                {
                    std::lock_guard<std::mutex> lk(state.mtx);
                    state.input      = input_img;
                    state.fused      = fused_img;
                    state.fps        = fps;
                    state.fps_smooth = fps_smooth;
                    state.detections = detections;
                    state.cpu_temp   = cpu_temp;
                    state.gpu_temp   = gpu_temp;
                    state.boxes      = std::move(boxes);
                }
            } catch (const nlohmann::json::exception&) {}
              catch (const cv::Exception&) {}
              catch (const std::bad_alloc& e) {
                  std::cerr << "[FETCH] bad_alloc: " << e.what() << " (payload size hint, skipping frame)" << std::endl;
              }
              catch (const std::exception& e) {
                  std::cerr << "[FETCH] exception: " << e.what() << std::endl;
              }
        }

        // 断开重连
        close(fd);
        { std::lock_guard<std::mutex> lk(state.mtx); state.connected = false; }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// ==========================================================================
// style
// ==========================================================================
static void setup_style()
{
    ImGui::StyleColorsDark();
    auto& s = ImGui::GetStyle();
    s.WindowRounding    = 4.0f;
    s.FrameRounding     = 3.0f;
    s.GrabRounding      = 3.0f;
    s.ScrollbarRounding = 3.0f;
    s.ChildRounding     = 4.0f;
    s.FramePadding      = ImVec2(6, 4);
    s.ItemSpacing       = ImVec2(6, 4);

    auto& c = s.Colors;
    c[ImGuiCol_Button]         = ImVec4(0.18f, 0.19f, 0.22f, 1.0f);
    c[ImGuiCol_ButtonHovered]  = ImVec4(0.25f, 0.27f, 0.31f, 1.0f);
    c[ImGuiCol_ButtonActive]   = ImVec4(0.20f, 0.21f, 0.26f, 1.0f);
    c[ImGuiCol_FrameBg]        = ImVec4(0.14f, 0.15f, 0.17f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.19f, 0.22f, 1.0f);
    c[ImGuiCol_WindowBg]       = ImVec4(0.10f, 0.11f, 0.12f, 1.0f);
    c[ImGuiCol_ChildBg]        = ImVec4(0.07f, 0.08f, 0.09f, 1.0f);
    c[ImGuiCol_TitleBg]        = ImVec4(0.12f, 0.13f, 0.15f, 1.0f);
    c[ImGuiCol_TitleBgActive]  = ImVec4(0.16f, 0.17f, 0.20f, 1.0f);
}

// ==========================================================================
// image panel
// ==========================================================================
static void image_panel(const char* title, GLTexture& tex, const cv::Mat& frame)
{
    if (!frame.empty()) tex.upload(frame);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    bool vis = ImGui::Begin(title);
    ImGui::PopStyleVar();
    if (vis) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        if (tex.get()) {
            float panel_aspect = avail.x / std::max(avail.y, 1.0f);
            float img_aspect   = tex.aspect();
            ImVec2 sz;
            if (panel_aspect > img_aspect) { sz.y = avail.y; sz.x = avail.y * img_aspect; }
            else                           { sz.x = avail.x; sz.y = avail.x / img_aspect; }
            float cx = (avail.x - sz.x) * 0.5f, cy = (avail.y - sz.y) * 0.5f;
            if (cx > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + cx);
            if (cy > 0) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + cy);
            ImGui::Image(tex.get(), sz);
        } else {
            ImVec2 ts = ImGui::CalcTextSize("Waiting for image...");
            ImGui::SetCursorPos(ImVec2((avail.x - ts.x)*0.5f, (avail.y - ts.y)*0.5f));
            ImGui::TextDisabled("Waiting for image...");
        }
    }
    ImGui::End();
}

// ==========================================================================
// arc gauge helper
// ==========================================================================
static void arc_gauge(const char* label, float value, float vmin, float vmax,
                      const ImVec4& col, float radius)
{
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 center = ImGui::GetCursorScreenPos() + ImVec2(radius, radius);

    float angle_min = 3.1416f * 0.75f;   // 135 deg
    float angle_max = 3.1416f * 2.25f;   // 405 deg

    // background arc
    dl->PathArcTo(center, radius, angle_min, angle_max, 32);
    dl->PathStroke(ImGui::GetColorU32(ImVec4(0.15f, 0.15f, 0.17f, 1.0f)), false, 6.0f);

    // value arc
    float t = std::clamp((value - vmin) / (vmax - vmin), 0.0f, 1.0f);
    float val_angle = angle_min + t * (angle_max - angle_min);
    dl->PathArcTo(center, radius, angle_min, val_angle, 32);
    dl->PathStroke(ImGui::GetColorU32(col), false, 6.0f);

    // tick marks + numbers
    float tick_step = (vmax - vmin) / 4.0f;
    for (int i = 0; i <= 4; ++i) {
        float vt = static_cast<float>(i) / 4.0f;
        float a = angle_min + vt * (angle_max - angle_min);
        float r1 = radius - 10, r2 = radius - 2;
        dl->PathLineTo(center + ImVec2(std::cos(a)*r1, std::sin(a)*r1));
        dl->PathLineTo(center + ImVec2(std::cos(a)*r2, std::sin(a)*r2));
        dl->PathStroke(ImGui::GetColorU32(ImVec4(0.4f, 0.4f, 0.4f, 1.0f)), false, 1.0f);
    }

    // value text in center
    char buf[16];
    snprintf(buf, sizeof(buf), "%.0f", value);
    ImVec2 ts = ImGui::CalcTextSize(buf);
    dl->AddText(center - ts * 0.5f, IM_COL32_WHITE, buf);

    // label below
    ImVec2 ls = ImGui::CalcTextSize(label);
    dl->AddText(center - ImVec2(ls.x * 0.5f, -radius + 8),
                ImGui::GetColorU32(ImVec4(0.6f, 0.6f, 0.6f, 1.0f)), label);

    ImGui::Dummy(ImVec2(radius * 2, radius * 2 + 20));
}

// ==========================================================================
// dashboard panel
// ==========================================================================
static void draw_dashboard(const DashboardCompute& dc, const RingBuffer<90>& det_buf,
                           float cpu_temp, float gpu_temp,
                           float jetson_fps, float display_fps)
{
    float panel_w = ImGui::GetContentRegionAvail().x - 8;

    // ---- mini histogram ----
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextUnformatted("Histogram");
    ImGui::PopFont();
    ImGui::Separator();

    float hist_h = 70;
    if (ImPlot::BeginPlot("##hist", ImVec2(panel_w, hist_h),
                          ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect |
                          ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxisLimits(ImAxis_X1, -0.5f, 31.5f, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.1f, ImGuiCond_Always);
        if (!dc.hist_bins.empty()) {
            static float xs[32], ys[32];
            for (int i = 0; i < 32; ++i) { xs[i] = static_cast<float>(i); ys[i] = dc.hist_bins[i]; }
            ImPlot::PlotBars("##histbars", xs, ys, 32, 0.9f);
        }
        ImPlot::EndPlot();
    }

    char info_buf[64];
    snprintf(info_buf, sizeof(info_buf), "mean %.0f  std %.0f", dc.hist_mean, dc.hist_stddev);
    ImGui::TextDisabled("%s", info_buf);

    ImGui::Spacing();

    // ---- motion / frame diff ----
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextUnformatted("Frame Diff");
    ImGui::PopFont();
    ImGui::Separator();

    ImVec4 motion_col = dc.motion_pct < 2.0f ? ImVec4(0.3f, 0.9f, 0.3f, 1.0f)
                      : dc.motion_pct < 8.0f ? ImVec4(0.9f, 0.7f, 0.2f, 1.0f)
                      :                        ImVec4(0.9f, 0.2f, 0.2f, 1.0f);
    ImGui::TextColored(motion_col, "%.1f%%", dc.motion_pct);

    // diff thumbnail rendering
    if (!dc.diff_thumb.empty()) {
        ImVec2 thumb_sz(panel_w * 0.9f, panel_w * 0.9f * 36.0f / 64.0f);
        ImVec2 p0 = ImGui::GetCursorScreenPos();

        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (int y = 0; y < dc.diff_thumb.rows; ++y) {
            for (int x = 0; x < dc.diff_thumb.cols; ++x) {
                float v = dc.diff_thumb.at<uchar>(y, x) / 255.0f;
                // heatmap: black(0) → blue → green → yellow → red → white(255)
                ImVec4 c;
                if (v < 0.25f)      c = ImVec4(0, 0, v*4, 1);
                else if (v < 0.5f)  c = ImVec4(0, (v-0.25f)*4, 1, 1);
                else if (v < 0.75f) c = ImVec4((v-0.5f)*4, 1, 1-(v-0.5f)*4, 1);
                else                c = ImVec4(1, 1-(v-0.75f)*4, 0, 1);

                float cx = p0.x + static_cast<float>(x) * thumb_sz.x / dc.diff_thumb.cols;
                float cy = p0.y + static_cast<float>(y) * thumb_sz.y / dc.diff_thumb.rows;
                float cw = thumb_sz.x / dc.diff_thumb.cols + 1;
                float ch = thumb_sz.y / dc.diff_thumb.rows + 1;
                dl->AddRectFilled(ImVec2(cx, cy), ImVec2(cx + cw, cy + ch),
                                  ImGui::GetColorU32(c));
            }
        }
        ImGui::Dummy(thumb_sz);
    }

    ImGui::Spacing();

    // ---- detection trend ----
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextUnformatted("Detections");
    ImGui::PopFont();
    ImGui::Separator();

    int dn = det_buf.count;
    if (dn >= 2) {
        std::vector<float> dvals(dn);
        det_buf.copy_chronological(dvals.data());

        float max_d = *std::max_element(dvals.begin(), dvals.end());
        float min_d = *std::min_element(dvals.begin(), dvals.end());
        float range = std::max(max_d - min_d, 1.0f);

        ImVec2 spark_sz(panel_w, 50);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 sp = ImGui::GetCursorScreenPos();

        // fill area
        for (int i = 0; i < dn - 1; ++i) {
            float x0 = sp.x + (static_cast<float>(i) / (dn - 1)) * spark_sz.x;
            float x1 = sp.x + (static_cast<float>(i + 1) / (dn - 1)) * spark_sz.x;
            float y0 = sp.y + (1.0f - (dvals[i] - min_d) / range) * spark_sz.y;
            float y1 = sp.y + (1.0f - (dvals[i + 1] - min_d) / range) * spark_sz.y;

            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, sp.y + spark_sz.y),
                              ImGui::GetColorU32(ImVec4(0.22f, 0.58f, 0.88f, 0.3f)));
        }
        // line
        for (int i = 0; i < dn - 1; ++i) {
            float x0 = sp.x + (static_cast<float>(i) / (dn - 1)) * spark_sz.x;
            float x1 = sp.x + (static_cast<float>(i + 1) / (dn - 1)) * spark_sz.x;
            float y0 = sp.y + (1.0f - (dvals[i] - min_d) / range) * spark_sz.y;
            float y1 = sp.y + (1.0f - (dvals[i + 1] - min_d) / range) * spark_sz.y;
            dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1),
                        ImGui::GetColorU32(ImVec4(0.22f, 0.58f, 0.88f, 1.0f)), 1.5f);
        }
        ImGui::Dummy(spark_sz);

        snprintf(info_buf, sizeof(info_buf), "current: %.0f  peak: %.0f", dvals.back(), max_d);
        ImGui::TextDisabled("%s", info_buf);
    } else {
        ImGui::TextDisabled("collecting...");
    }

    ImGui::Spacing();

    // ---- temperature gauges ----
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextUnformatted("Temperature");
    ImGui::PopFont();
    ImGui::Separator();

    float gauge_r = (panel_w - 12) * 0.5f;
    if (cpu_temp >= 0) arc_gauge("CPU", cpu_temp, 20, 100, ImVec4(0.9f, 0.65f, 0.15f, 1.0f), gauge_r * 0.48f);
    ImGui::SameLine();
    if (gpu_temp >= 0) arc_gauge("GPU", gpu_temp, 20, 100, ImVec4(0.9f, 0.25f, 0.25f, 1.0f), gauge_r * 0.48f);

    ImGui::Spacing();

    // ---- FPS gauges ----
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextUnformatted("Frame Rate");
    ImGui::PopFont();
    ImGui::Separator();

    if (jetson_fps >= 0) arc_gauge("Jetson", jetson_fps, 0, 60, ImVec4(0.3f, 0.9f, 0.3f, 1.0f), gauge_r * 0.48f);
    ImGui::SameLine();
    if (display_fps >= 0) arc_gauge("Display", display_fps, 0, 60, ImVec4(0.3f, 0.5f, 0.9f, 1.0f), gauge_r * 0.48f);
}

// ==========================================================================
// bottom charts (always visible)
// ==========================================================================
static void draw_charts(
    const RingBuffer<300>& fps_raw, const RingBuffer<300>& fps_smooth,
    const RingBuffer<300>& fps_disp,
    const RingBuffer<300>& cpu_buf, const RingBuffer<300>& gpu_buf,
    float elapsed)
{
    int n = fps_raw.count;
    if (n < 2) { ImGui::TextDisabled("Collecting data..."); return; }

    std::vector<float> t_axis(n);
    std::vector<float> raw(n), smooth(n), disp(n), cpu(n), gpu(n);
    fps_raw.copy_chronological(raw.data());
    fps_smooth.copy_chronological(smooth.data());
    fps_disp.copy_chronological(disp.data());
    cpu_buf.copy_chronological(cpu.data());
    gpu_buf.copy_chronological(gpu.data());

    float dt = 0.25f;
    for (int i = 0; i < n; ++i) t_axis[i] = elapsed - (n - 1 - i) * dt;

    float avail_h = ImGui::GetContentRegionAvail().y;
    float chart_h = (avail_h - ImGui::GetStyle().ItemSpacing.y) * 0.47f;

    // FPS
    if (ImPlot::BeginPlot("##fps_plot", ImVec2(-1, chart_h),
                          ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {
        ImPlot::SetupAxis(ImAxis_X1, "Time (s)", ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y1, "FPS", ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, std::max(0.0f, elapsed - 75.0f), elapsed + 2, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100);

        ImPlot::SetNextLineStyle(ImVec4(0.3f, 0.7f, 0.9f, 1.0f), 2.0f);
        ImPlot::PlotLine("Jetson FPS", t_axis.data(), raw.data(), n);
        ImPlot::SetNextLineStyle(ImVec4(0.3f, 0.9f, 0.5f, 1.0f), 1.5f);
        ImPlot::PlotLine("Jetson Avg", t_axis.data(), smooth.data(), n);
        ImPlot::SetNextLineStyle(ImVec4(0.9f, 0.7f, 0.2f, 1.0f), 1.5f);
        ImPlot::PlotLine("Display", t_axis.data(), disp.data(), n);
        ImPlot::EndPlot();
    }

    // Temperature
    if (ImPlot::BeginPlot("##temp_plot", ImVec2(-1, chart_h),
                          ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {
        ImPlot::SetupAxis(ImAxis_X1, "Time (s)", ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y1, "Temp (C)", ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, std::max(0.0f, elapsed - 75.0f), elapsed + 2, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 20, 100);

        ImPlot::SetNextLineStyle(ImVec4(0.9f, 0.65f, 0.15f, 1.0f), 2.0f);
        ImPlot::PlotLine("CPU", t_axis.data(), cpu.data(), n);
        ImPlot::SetNextLineStyle(ImVec4(0.9f, 0.25f, 0.25f, 1.0f), 2.0f);
        ImPlot::PlotLine("GPU", t_axis.data(), gpu.data(), n);

        double wy = 80.0;
        ImPlot::DragLineY(0, &wy, ImVec4(0.9f, 0.2f, 0.2f, 0.5f), 1.0f, ImPlotDragToolFlags_NoInputs);
        ImPlot::EndPlot();
    }
}

// ==========================================================================
// status bar
// ==========================================================================
static void draw_status(int fps, float fps_smooth, int detections,
                        float display_fps, float uptime_s)
{
    int m = static_cast<int>(uptime_s) / 60;
    int s = static_cast<int>(uptime_s) % 60;
    ImGui::Text("Jetson FPS: %d (avg %.0f)  |  Detections: %d  |  Uptime: %dm %02ds",
                fps, fps_smooth, detections, m, s);
    ImGui::SameLine(ImGui::GetWindowWidth() - 140);
    ImGui::Text("Display: %.0f FPS", display_fps);
}

// ==========================================================================
// main
// ==========================================================================
int main(int argc, char* argv[])
{
    AppState state;
    if (argc >= 2) {
        state.host = argv[1];
        auto pos = state.host.find("://");
        if (pos != std::string::npos) state.host = state.host.substr(pos + 3);
        if (state.host.find(':') == std::string::npos) state.host += ":8080";
    }

    // ---- GLFW + OpenGL ----
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1800, 950, "Particle Viewer", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ---- ImGui + ImPlot ----
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;

    float font_size = 18.0f;
    io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size);
    ImGui::GetStyle().ScaleAllSizes(font_size / 13.0f);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    setup_style();

    // ---- fetch ----
    std::atomic<bool> stop{false};
    std::thread fetcher(fetch_thread, std::ref(state), std::ref(stop));
    fetcher.detach();

    // ---- textures ----
    GLTexture tex_input, tex_fused;

    // ---- display FPS ----
    auto t_last = std::chrono::steady_clock::now();
    auto t_start = std::chrono::steady_clock::now();
    float display_fps = 0.0f;
    int frame_count = 0;

    // ---- host buffer ----
    char host_buf[256] = {};
    { std::lock_guard<std::mutex> lk(state.mtx); snprintf(host_buf, sizeof(host_buf), "%s", state.host.c_str()); }

    // ---- split ratios ----
    float split_ratio = 0.45f;
    bool show_dashboard = true;

    // ---- time-series buffers ----
    RingBuffer<300> buf_fps_raw, buf_fps_smooth, buf_fps_disp, buf_cpu, buf_gpu;
    RingBuffer<90>  buf_detections;   // ~90s at 1Hz
    auto t_last_sample = std::chrono::steady_clock::now();
    auto t_last_det_sample = std::chrono::steady_clock::now();

    // ---- dashboard compute ----
    DashboardCompute dash_comp;
    int last_output_gen = -1;  // track output frame changes

    // ---- main loop ----
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Tab toggles dashboard
        if (ImGui::IsKeyPressed(ImGuiKey_Tab)) show_dashboard = !show_dashboard;

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        ImGuiWindowFlags root_flags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_MenuBar;

        ImGui::Begin("##root", nullptr, root_flags);
        ImGui::PopStyleVar(2);

        // ---- menu bar ----
        if (ImGui::BeginMenuBar()) {
            bool conn;
            { std::lock_guard<std::mutex> lk(state.mtx); conn = state.connected; }

            ImVec4 dot_c = conn ? ImVec4(0.2f, 0.9f, 0.3f, 1.0f) : ImVec4(0.9f, 0.3f, 0.2f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, dot_c);
            ImGui::TextUnformatted(conn ? " \xe2\x97\x8f  ONLINE" : " \xe2\x97\x8f  OFFLINE");
            ImGui::PopStyleColor();

            ImGui::SameLine();
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
            ImGui::SameLine();

            ImGui::SetNextItemWidth(170);
            if (ImGui::InputTextWithHint("##host", "host:port", host_buf, sizeof(host_buf),
                                         ImGuiInputTextFlags_EnterReturnsTrue)) {
                std::lock_guard<std::mutex> lk(state.mtx);
                state.host = host_buf; state.connected = false; state.connecting = true;
                stop.store(true); std::this_thread::sleep_for(std::chrono::milliseconds(200));
                stop.store(false);
                std::thread t(fetch_thread, std::ref(state), std::ref(stop)); t.detach();
            }
            ImGui::SameLine();
            if (ImGui::Button("Connect")) {
                std::lock_guard<std::mutex> lk(state.mtx);
                state.host = host_buf; state.connected = false; state.connecting = true;
                stop.store(true); std::this_thread::sleep_for(std::chrono::milliseconds(200));
                stop.store(false);
                std::thread t(fetch_thread, std::ref(state), std::ref(stop)); t.detach();
            }
            bool connecting;
            { std::lock_guard<std::mutex> lk(state.mtx); connecting = state.connecting; }
            if (connecting) { ImGui::SameLine(); ImGui::TextDisabled("connecting..."); }

            ImGui::SameLine(ImGui::GetWindowWidth() - 200);
            ImGui::TextDisabled("Tab: panel  |  %.0f FPS", display_fps);
            ImGui::EndMenuBar();
        }

        // ---- copy state ----
        cv::Mat input_frame, fused_frame;
        int fps, detections;
        float fps_smooth, cpu_temp, gpu_temp;
        std::vector<BoxInfo> boxes;
        {
            std::lock_guard<std::mutex> lk(state.mtx);
            input_frame = state.input;
            fused_frame = state.fused;
            fps         = state.fps;
            fps_smooth  = state.fps_smooth;
            detections  = state.detections;
            cpu_temp    = state.cpu_temp;
            gpu_temp    = state.gpu_temp;
            boxes       = state.boxes;
        }

        // ---- dashboard compute (lazy: only when fused changes) ----
        if (!fused_frame.empty()) {
            const uchar* ptr = fused_frame.ptr();
            int gen = static_cast<int>(reinterpret_cast<uintptr_t>(ptr));
            if (gen != last_output_gen) {
                dash_comp.update(fused_frame);
                last_output_gen = gen;
            }
        }

        // ---- time-series sampling ----
        auto t_now = std::chrono::steady_clock::now();
        float s_dt = std::chrono::duration<float>(t_now - t_last_sample).count();
        if (s_dt >= 0.25f && frame_count > 0) {
            buf_fps_raw.push(static_cast<float>(fps));
            buf_fps_smooth.push(fps_smooth);
            buf_fps_disp.push(display_fps);
            buf_cpu.push(cpu_temp);
            buf_gpu.push(gpu_temp);
            t_last_sample = t_now;
        }
        float det_dt = std::chrono::duration<float>(t_now - t_last_det_sample).count();
        if (det_dt >= 1.0f && frame_count > 0) {
            buf_detections.push(static_cast<float>(detections));
            t_last_det_sample = t_now;
        }

        // ---- content layout ----
        ImVec2 content_size = ImGui::GetContentRegionAvail();
        float bar_h = content_size.y * 0.37f;
        float top_h = content_size.y - bar_h - ImGui::GetStyle().ItemSpacing.y;

        // dashboard width
        float dash_w = show_dashboard ? 280.0f : 0.0f;
        float dash_w_total = show_dashboard ? dash_w + ImGui::GetStyle().ItemSpacing.x : 0.0f;
        float images_w = content_size.x - dash_w_total;
        float left_w = images_w * split_ratio - 4.0f;

        // ---- top row: images + dashboard ----
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.05f, 0.06f, 0.07f, 1.0f));

        // left: input + boxes
        ImGui::BeginChild("##left", ImVec2(left_w, top_h), ImGuiChildFlags_Border);
        image_panel("Input", tex_input, input_frame);
        ImGui::EndChild();
        ImGui::SameLine();

        // splitter
        {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.27f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.32f, 0.32f, 0.35f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.35f, 0.35f, 0.38f, 1.0f));
            ImGui::Button("##splitter", ImVec2(8.0f, top_h));
            if (ImGui::IsItemActive())
                split_ratio += ImGui::GetIO().MouseDelta.x / images_w;
            split_ratio = std::clamp(split_ratio, 0.15f, 0.85f);
            ImGui::PopStyleColor(3);
        }
        ImGui::SameLine();

        // right: fused
        float right_w = images_w - left_w - 12.0f;
        ImGui::BeginChild("##right", ImVec2(right_w, top_h), ImGuiChildFlags_Border);
        image_panel("Fused", tex_fused, fused_frame);
        ImGui::EndChild();

        // dashboard
        if (show_dashboard) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.11f, 0.12f, 1.0f));
            ImGui::BeginChild("##dashboard", ImVec2(0, top_h), ImGuiChildFlags_Border);
            draw_dashboard(dash_comp, buf_detections, cpu_temp, gpu_temp, fps_smooth, display_fps);
            ImGui::EndChild();
            ImGui::PopStyleColor();
        }

        ImGui::PopStyleColor();
        ImGui::PopStyleVar();

        // ---- bottom: charts + status ----
        {
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.11f, 0.12f, 1.0f));
            ImGui::BeginChild("##bottom", ImVec2(0, 0), ImGuiChildFlags_Border);
            ImGui::PopStyleColor();
            ImGui::PopStyleVar();

            float elapsed = std::chrono::duration<float>(t_now - t_start).count();

            // charts fill most of bottom area
            float status_h = ImGui::GetFrameHeightWithSpacing();
            ImGui::BeginChild("##charts_area", ImVec2(0, -status_h), ImGuiChildFlags_None);
            draw_charts(buf_fps_raw, buf_fps_smooth, buf_fps_disp, buf_cpu, buf_gpu, elapsed);
            ImGui::EndChild();

            ImGui::Separator();
            draw_status(fps, fps_smooth, detections, display_fps, elapsed);

            ImGui::EndChild();
        }

        ImGui::End(); // root

        // ---- render ----
        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.08f, 0.09f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);

        // display FPS
        float dt = std::chrono::duration<float>(t_now - t_last).count();
        display_fps = display_fps * 0.9f + (1.0f / std::max(dt, 0.001f)) * 0.1f;
        t_last = t_now;
        frame_count++;
    }

    // ---- cleanup ----
    stop.store(true);
    tex_input.destroy();
    tex_fused.destroy();
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
