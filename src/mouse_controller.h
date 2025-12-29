#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <string>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>

#include "fa.h"

// 只保留“最初版本”的鼠标控制参数（吸附 + 计划式拆步），删除三段式动态策略参数。
struct MouseControllerConfig {
    // Allow relative movement (pixels). attachTo/planTick use relative movement.
    bool allowRelative = true;

    // Optional: require a modifier key held to allow movement (0 disables).
    int enableKeyVk = 0;

    // Foreground requirement (optional)
    HWND requiredForeground = nullptr;

    // --- 近距离吸附（legacy）参数 ---
    int   attachDeadzonePx = 1;     // within this distance, stop moving
    float attachGain = 1.6f;        // gain on error
    int   attachMaxStepPx = 60;     // max pixels per tick/frame (clamped to <=127)
    float attachSmoothing = 1.0f;   // 0..1 (higher = snappier)

    // --- STM32 HID Feature output (FaController) ---
    bool useFaHid = true;
    stm32hid::FaConfig fa{};
    bool faSendEnableDisable = true;
    bool faActive = false;        // backend 同步：enabled && attach
    int  faReconnectMs = 1000;
};

class MouseController {
public:
    MouseController() = default;

    bool initialize(const MouseControllerConfig& cfg = {});
    void shutdown();
    bool isInitialized() const { return m_initialized; }

    void updateConfig(const MouseControllerConfig& cfg);
    void setConfig(const MouseControllerConfig& cfg) { updateConfig(cfg); }
    const MouseControllerConfig& config() const { return m_cfg; }

    // --- UI 调试读取 ---
    void getLastMove(int& dx, int& dy) const {
        dx = m_lastDx.load(std::memory_order_relaxed);
        dy = m_lastDy.load(std::memory_order_relaxed);
    }
    bool faConnected() const { return m_fa.is_open(); }
    bool faActive() const { return m_faWasActive; }

    // ---- UI 调试：最后一次输出的 dx/dy ----
    std::atomic<int> m_lastDx{ 0 };
    std::atomic<int> m_lastDy{ 0 };

    // ✅ 只保留硬件相对移动
    bool moveRelative(int dx, int dy);

    // 旧接口保留（内部仍走硬件相对，做一次吸附）
    bool attachTo(int targetX, int targetY);

    // 识别帧到来：生成“计划式拆步”移动计划
    void planNewTarget(int targetX, int targetY,
        float framePeriodMs, float moveTickMs, float alpha);

    // 控制 tick：执行计划的一步（每 move_tick 调一次）
    bool planTick();

    void planReset();
    void resetAttach();

    const std::string& lastError() const { return m_lastError; }
    static bool isKeyDown(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

private:
    bool allowMoveByConfig() const;
    void setLastError(const char* what);

    MouseControllerConfig m_cfg{};
    bool m_initialized{ false };
    std::string m_lastError;

    // legacy attach smoothing state
    float m_smoothDx{ 0.0f };
    float m_smoothDy{ 0.0f };

    // HID controller
    stm32hid::FaController m_fa;
    bool m_faWasActive = false;
    std::chrono::steady_clock::time_point m_nextFaReconnect{};

    // --- planned move (legacy N-step) ---
    float m_planRemDx{ 0.0f };
    float m_planRemDy{ 0.0f };
    int   m_planStepsLeft{ 0 };

    // residual accumulation to fight int8 quantization
    float m_planFracDx{ 0.0f };
    float m_planFracDy{ 0.0f };

    //// --- new: target-based plan (no N-step) ---
    //bool  m_planHasTarget{ false };
    //float m_planTargetX{ 0.0f };
    //float m_planTargetY{ 0.0f };

};
