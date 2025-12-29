#pragma once
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>

#include "tracking.h"  // EyeTrackerState / TargetTrack

// 前置声明（减少 include 依赖）
struct RuntimeParams;
struct RoiInfo;
struct Det;

class DxgiDup;
class TrtRunner;
class MouseController;

class BackendWorker {
public:
    explicit BackendWorker(RuntimeParams* params);
    ~BackendWorker();

    bool start();
    void stop();

    // --- UI 调试统计快照（线程安全读取）---
    struct UiStats {
        double cap_fps = 0.0;
        double infer_fps = 0.0;
        int det_count = 0;
        int track_count = 0;
        int locked_uid = 0;
        bool lock_has = false;
        bool lock_residual = false;
        int lock_uid = 0;
        int lock_miss = 0;
        int lock_srank = -1;
        int dx = 0;
        int dy = 0;
        bool fa_connected = false;
        bool fa_active = false;
        long cap_last_hr = 0;      // HRESULT
        int cap_fail_streak = 0;
        int cap_recover_count = 0;
        int cap_black_streak = 0;
        bool cap_permission_hint = false;
    };
    UiStats getUiStats() const {
    std::lock_guard<std::mutex> lk(m_uiMu);
    return m_uiStats;
}

    bool m_wasDown = false;

    void setEnabled(bool enabled);
    void requestRestart();
    void requestClear();

private:
    bool initResources();
    void runLoop();
    void cleanup();

    // ✅ 双缓冲后：不再传 host_out
    void processFrame(int roi, bool attach);

    void updateTracking(const std::vector<Det>& dets, const RoiInfo& info, bool attach); // 占位
    bool safeAttach(MouseController& mouse, TargetTrack& tr, float framePeriodMs);

private:
    RuntimeParams* m_params = nullptr;

    std::atomic<bool> m_running{ false };
    std::thread m_thread;

    // --- UI stats（worker 写、UI 读）---
    mutable std::mutex m_uiMu;
    UiStats m_uiStats{};
    uint64_t m_capFrames = 0;
    uint64_t m_inferFrames = 0;
    std::chrono::steady_clock::time_point m_capT0{};
    std::chrono::steady_clock::time_point m_inferT0{};

    // 资源
    std::unique_ptr<DxgiDup>         m_dup;
    std::unique_ptr<TrtRunner>       m_trt;
    std::unique_ptr<MouseController> m_mouse;

    // CUDA interop
    void* m_stream = nullptr;   // cudaStream_t（cpp 里强转）
    void* m_cuRes = nullptr;   // cudaGraphicsResource*
    void* m_regTex = nullptr;   // ID3D11Texture2D*（仅用于比较指针）

    // ---------- ✅ 双缓冲：Host 输出 + Event ----------
    struct HostOutSlot {
        float* host = nullptr;  // pinned host buffer
        void* ev = nullptr;  // cudaEvent_t（用 void* 避免头文件 include cuda）
        bool   pending = false;
        uint64_t seq = 0;

        // 元数据快照
        int roi = 0;
        int ox = 0, oy = 0, full_w = 0, full_h = 0;
        bool attach = false;
        float conf_thres = 0.0f;
        float iou_thres = 0.0f;
    };

    HostOutSlot m_outSlots[2]{};
    size_t   m_outBytes = 0;
    uint64_t m_outSeq = 0;

    // 多目标轨迹池 + 当前锁定目标（可能 residual）
    EyeTrackerState m_tracker{};
    TargetTrack     m_lock{};
};

