#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "mouse_controller.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

static inline int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}
static inline float clampf(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}
static inline float clamp01(float v) { return clampf(v, 0.0f, 1.0f); }

bool MouseController::initialize(const MouseControllerConfig& cfg) {
    updateConfig(cfg);
    m_initialized = true;
    return true;
}

void MouseController::shutdown() {
    planReset();

    if (m_fa.is_open()) {
        if (m_cfg.faSendEnableDisable) (void)m_fa.send_enable(false);
        m_fa.close();
    }
    m_faWasActive = false;
    m_initialized = false;
}

void MouseController::setLastError(const char* what) {
    m_lastError = what ? what : "";
}

bool MouseController::allowMoveByConfig() const {
    if (!m_initialized) return false;

    // 已撤掉 WinAPI 移动：必须启用硬件 HID。
    if (!m_cfg.useFaHid) return false;

    // 前台窗口要求（可选）
    if (m_cfg.requiredForeground != nullptr) {
        HWND fg = GetForegroundWindow();
        if (fg != m_cfg.requiredForeground) return false;
    }

    // 按键门控（可选）
    if (m_cfg.enableKeyVk != 0) {
        if ((GetAsyncKeyState(m_cfg.enableKeyVk) & 0x8000) == 0) return false;
    }

    // HID 没打开就不移动（避免丢命令）
    if (!m_fa.is_open()) return false;

    // backend 会把 faActive 同步成 enabled && attach；这里再兜底 gate
    if (!m_cfg.faActive) return false;

    return true;
}

void MouseController::updateConfig(const MouseControllerConfig& cfg) {
    m_cfg = cfg;

    // ----------------- sanitize：只保留“最初版本”那组参数 -----------------
    m_cfg.attachDeadzonePx = std::max(0, m_cfg.attachDeadzonePx);
    m_cfg.attachGain = std::max(0.0f, m_cfg.attachGain);
    // 单 tick 最大步长：强制 <=127，避免 int8 拆包造成“瞬跳”
    m_cfg.attachMaxStepPx = clampi(m_cfg.attachMaxStepPx, 1, 127);
    m_cfg.attachSmoothing = clamp01(m_cfg.attachSmoothing);
    m_cfg.faReconnectMs = std::max(200, m_cfg.faReconnectMs);

    // ----------------- HID 生命周期管理 -----------------
    if (!m_cfg.useFaHid) {
        if (m_fa.is_open()) {
            if (m_cfg.faSendEnableDisable) (void)m_fa.send_enable(false);
            m_fa.close();
        }
        m_faWasActive = false;
        return;
    }

    const auto now = std::chrono::steady_clock::now();

    // 需要时尝试重连
    if (!m_fa.is_open() && now >= m_nextFaReconnect) {
        if (!m_fa.open(m_cfg.fa)) {
            m_nextFaReconnect = now + std::chrono::milliseconds(m_cfg.faReconnectMs);
            m_faWasActive = false;
        }
        else {
            m_nextFaReconnect = now;
            m_faWasActive = false; // 重新打开后重新走 enable 边沿
        }
    }

    // enable/disable 边沿发送
    if (m_cfg.faSendEnableDisable) {
        const bool wantActive = m_cfg.faActive;
        if (wantActive != m_faWasActive) {
            if (m_fa.is_open()) (void)m_fa.send_enable(wantActive);
            m_faWasActive = wantActive;
        }
    }
    else {
        if (!m_fa.is_open()) m_faWasActive = false;
    }
}

bool MouseController::moveRelative(int dx, int dy) {
    if (!allowMoveByConfig()) return false;
    if (!m_cfg.allowRelative) {
        setLastError("Relative move disallowed by config");
        return false;
    }

    // --- UI debug: record last requested move ---
    m_lastDx.store(dx, std::memory_order_relaxed);
    m_lastDy.store(dy, std::memory_order_relaxed);

    int rx = dx;
    int ry = dy;

    while (rx != 0 || ry != 0) {
        const int sx = clampi(rx, -127, 127);
        const int sy = clampi(ry, -127, 127);

        if (!m_fa.is_open()) { setLastError("FaHID not open"); return false; }
        if (!m_fa.send_move_once((int8_t)sx, (int8_t)sy)) { setLastError("FaHID send_move failed"); return false; }

        rx -= sx;
        ry -= sy;
    }
    return true;
}


void MouseController::planNewTarget(int targetX, int targetY,
    float /*framePeriodMs*/, float /*moveTickMs*/, float alpha)
{
    if (!allowMoveByConfig()) { planReset(); return; }

    POINT p{};
    if (!GetCursorPos(&p)) return;

    float ex = (float)targetX - (float)p.x;
    float ey = (float)targetY - (float)p.y;

    alpha = std::clamp(alpha, 0.0f, 1.0f);

    // rem 融合：连续且不抖
    m_planRemDx = alpha * m_planRemDx + (1.0f - alpha) * ex;
    m_planRemDy = alpha * m_planRemDy + (1.0f - alpha) * ey;
}




void MouseController::planReset()
{
    m_planRemDx = 0.0f;
    m_planRemDy = 0.0f;
    m_planFracDx = 0.0f;
    m_planFracDy = 0.0f;
    m_planStepsLeft = 0;
}



void MouseController::resetAttach() {
    m_smoothDx = 0.0f;
    m_smoothDy = 0.0f;
}

static float Smoothstep(float a, float b, float x) {
    float t = (x - a) / (b - a);
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

bool MouseController::planTick()
{
    if (!allowMoveByConfig()) return false;

    const float arrive = std::max(0.5f, (float)m_cfg.attachDeadzonePx);

    // 到位直接停
    if (std::abs(m_planRemDx) <= arrive && std::abs(m_planRemDy) <= arrive) {
        planReset();
        return true;
    }

    // ===== 1) 计算动态 maxStepEff（远大近小，连续过渡）=====
    int nearStep = std::max(20, (int)std::lround(0.5f * (float)m_cfg.attachMaxStepPx));     // 近点精细
    int farStep = std::clamp(m_cfg.attachMaxStepPx, 127, 127);   // 远点速度

    float dist = std::max(std::abs(m_planRemDx), std::abs(m_planRemDy));
    float t = Smoothstep(20.0f, 50.0f, dist);                  // 12~120px 过渡
    int maxStepEff = (int)std::lround(nearStep + t * (farStep - nearStep));
    maxStepEff = std::clamp(maxStepEff, 1, 127);

    // ===== 2) N 由“距离/最大步长”决定（不跟 fps 耦合）=====
    int N = (int)std::ceil(dist / (float)maxStepEff);
    N = std::clamp(N, 1, 2);

    // ===== 3) 平均拆步 + 分数累积抗量化 =====
    float sx = (m_planRemDx / (float)N) + m_planFracDx;
    float sy = (m_planRemDy / (float)N) + m_planFracDy;

    int ix = (int)std::lround(sx);
    int iy = (int)std::lround(sy);

    // 量化成 0 时，给 1 像素避免卡住
    if (ix == 0 && std::abs(m_planRemDx) > arrive) ix = (m_planRemDx > 0 ? 1 : -1);
    if (iy == 0 && std::abs(m_planRemDy) > arrive) iy = (m_planRemDy > 0 ? 1 : -1);

    // 限幅到动态上限
    ix = std::clamp(ix, -maxStepEff, maxStepEff);
    iy = std::clamp(iy, -maxStepEff, maxStepEff);

    if (!moveRelative(ix, iy)) return false;

    // ===== 4) 用“实际发送步长”更新剩余（稳定关键，不绕圈）=====
    m_planRemDx -= (float)ix;
    m_planRemDy -= (float)iy;

    m_planFracDx = sx - (float)ix;
    m_planFracDy = sy - (float)iy;

    return true;
}



