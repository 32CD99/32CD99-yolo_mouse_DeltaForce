#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <wrl/client.h>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <thread>
#include <filesystem>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

#include "backend_worker.h"
#include "runtime_params.h"
#include "tracking.h"

#include "dxgi_dup.h"
#include "trt_runner.h"
#include "preprocess.h"
#include "postprocess.h"
#include "mouse_controller.h"

#include <sstream>
#include <fstream>






static std::wstring HrToWString(HRESULT hr) {
    wchar_t* msg = nullptr;
    DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;
    FormatMessageW(flags, nullptr, (DWORD)hr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPWSTR)&msg, 0, nullptr);
    std::wstringstream ss;
    ss << L"0x" << std::hex << (unsigned long)hr;
    if (msg) {
        ss << L" (" << msg << L")";
        LocalFree(msg);
    }
    return ss.str();
}

static void AppendLog(const std::wstring& line) {
    // 写到当前工作目录下（你也可以改成 exe 同目录）
    std::wofstream f(L"dd_trt_yolo.log", std::ios::app);
    if (!f) return;
    f << line << L"\n";
}

static void PopupFail(const wchar_t* step, const std::wstring& detail) {
    std::wstring msg = L"[BackendWorker start failed]\n\nSTEP: ";
    msg += step;
    msg += L"\n\nDETAIL:\n";
    msg += detail;
    msg += L"\n\n(同时已写入 dd_trt_yolo.log)";
    AppendLog(msg);
    MessageBoxW(nullptr, msg.c_str(), L"dd_trt_yolo", MB_ICONERROR | MB_OK);
}








// 发布版关调试输出（你也可以改成读取配置）
static constexpr bool DEBUG_ENABLED = false;

// FPS cap steps (scheme B)
// F6 cycles: 60 -> 90 -> 120 -> 144 -> ...
static int SanitizeFpsCap(int v) {
    constexpr int kSteps[] = { 60, 90, 120, 144 };
    int best = kSteps[0];
    int bestAbs = abs(v - best);
    for (int s : kSteps) {
        const int d = abs(v - s);
        if (d < bestAbs) { bestAbs = d; best = s; }
    }
    return best;
}

static int NextFpsCap(int v) {
    constexpr int kSteps[] = { 60, 90, 120, 144 };
    for (int i = 0; i < 4; ++i) {
        if (v == kSteps[i]) return kSteps[(i + 1) & 3];
    }
    // 如果 v 不是整档，先贴近再下一档
    const int snapped = SanitizeFpsCap(v);
    for (int i = 0; i < 4; ++i) {
        if (snapped == kSteps[i]) return kSteps[(i + 1) & 3];
    }
    return kSteps[0];
}

// Resolve engine path relative to the executable (engine placed next to .exe)
static std::filesystem::path GetExeDir() {
    wchar_t buf[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::filesystem::path p(buf, buf + n);
    return p.parent_path();
}

using Microsoft::WRL::ComPtr;

static bool CheckCuda(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    if (DEBUG_ENABLED) {
        OutputDebugStringA("[CUDA] ");
        OutputDebugStringA(what);
        OutputDebugStringA(" : ");
        OutputDebugStringA(cudaGetErrorString(e));
        OutputDebugStringA("\n");
    }
    return false;
}

BackendWorker::BackendWorker(RuntimeParams* params) : m_params(params) {}

BackendWorker::~BackendWorker() {
    stop();
}

bool BackendWorker::start()
{
    if (m_running.load()) return true;

    try {
        if (!initResources()) {
            // initResources 内部会 PopupFail（我们下面会加）
            return false;
        }
    }
    catch (const std::exception& e) {
        std::wstring w = L"std::exception: ";
        w += std::wstring(e.what(), e.what() + strlen(e.what()));
        PopupFail(L"exception", w);
        return false;
    }
    catch (...) {
        PopupFail(L"exception", L"unknown exception");
        return false;
    }

    m_running.store(true);
    m_thread = std::thread(&BackendWorker::runLoop, this);
    return true;
}


void BackendWorker::stop() {
    m_running.store(false);
    if (m_thread.joinable()) m_thread.join();
    cleanup();
}

void BackendWorker::setEnabled(bool enabled) {
    if (!m_params) return;
    m_params->enabled.store(enabled);

    if (m_mouse) {
        MouseControllerConfig cfg = m_mouse->config();
        cfg.faActive = false;
        m_mouse->updateConfig(cfg);
        m_mouse->resetAttach();
        m_mouse->planReset();
    }

    if (!enabled) {
        // ✅ 关闭时：彻底清理多目标池与锁定（干净）
        m_tracker = EyeTrackerState{};
        m_lock = TargetTrack{};
    }
}

void BackendWorker::requestRestart() {
    if (m_params) m_params->reqRestart.store(true);
}

void BackendWorker::requestClear() {
    if (m_params) m_params->reqClear.store(true);
}

bool BackendWorker::initResources() {
    auto Fail = [&](const wchar_t* step, const std::wstring& msg) -> bool {





        MessageBoxW(nullptr, (std::wstring(L"[start failed] ") + step + L"\n\n" + msg).c_str(),
            L"dd_trt_yolo", MB_ICONERROR | MB_OK);






        std::wstring out = L"[initResources FAIL] ";
        out += step;
        out += L"\n";
        out += msg;
        out += L"\n";
        OutputDebugStringW(out.c_str());
        return false;
        };

    // 0) 清理旧资源（防止重复 start/重启时残留）
    cleanup();  // 如果你已有 cleanup() 就用；没有就先删掉这行

    // 1) DXGI
    m_dup = std::make_unique<DxgiDup>();
    if (!m_dup->init(0)) {
        return Fail(L"DXGI init", L"m_dup->init(0) failed.");
    }
    if (!m_dup->adapter()) {
        return Fail(L"DXGI adapter", L"m_dup->adapter() is null.");
    }

    // 2) CUDA device from D3D11 adapter
    int cudaDev = 0;
    cudaError_t ce = cudaD3D11GetDevice(&cudaDev, m_dup->adapter());
    if (ce != cudaSuccess) {
        std::wstring msg = L"cudaD3D11GetDevice failed: ";
        msg += std::wstring(cudaGetErrorString(ce), cudaGetErrorString(ce) + strlen(cudaGetErrorString(ce)));
        return Fail(L"cudaD3D11GetDevice", msg);
    }

    if (!CheckCuda(cudaSetDevice(cudaDev), "cudaSetDevice")) {
        return Fail(L"cudaSetDevice", L"CheckCuda(cudaSetDevice) failed.");
    }

    // 3) Stream
    cudaStream_t stream{};
    if (!CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        return Fail(L"cudaStreamCreate", L"CheckCuda(cudaStreamCreate) failed.");
    }
    m_stream = (void*)stream;

    // 4) TRT engine: exe 同目录
    const std::filesystem::path enginePath = GetExeDir() / L"416_fp16.engine";
    if (!std::filesystem::exists(enginePath)) {
        return Fail(L"enginePath",
            L"engine not found: " + enginePath.wstring());
    }

    m_trt = std::make_unique<TrtRunner>();
    if (!m_trt->load_engine(enginePath.u8string())) {
        return Fail(L"TrtRunner::load_engine",
            L"load_engine failed. Engine: " + enginePath.wstring() +
            L"\n(常见：TRT版本不匹配 / 缺DLL：nvinfer_plugin,cublasLt,cudart...)");
    }

    // 5) 双缓冲：pinned host + event
    m_outBytes = m_trt->output_elems() * sizeof(float);
    for (int i = 0; i < 2; ++i) {
        m_outSlots[i] = HostOutSlot{};
        if (!CheckCuda(cudaHostAlloc((void**)&m_outSlots[i].host, m_outBytes, cudaHostAllocPortable),
            "cudaHostAlloc(host_out_slot)")) {
            return Fail(L"cudaHostAlloc", L"cudaHostAlloc for slot failed.");
        }
        cudaEvent_t ev{};
        if (!CheckCuda(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
            "cudaEventCreateWithFlags")) {
            return Fail(L"cudaEventCreateWithFlags", L"event create failed.");
        }
        m_outSlots[i].ev = (void*)ev;
        m_outSlots[i].pending = false;
        m_outSlots[i].seq = 0;
    }
    m_outSeq = 0;

    // 6) Mouse
    m_mouse = std::make_unique<MouseController>();
    MouseControllerConfig mcfg{};

    mcfg.attachDeadzonePx = m_params->attachDeadzonePx.load();
    mcfg.attachGain = m_params->attachGain.load();
    mcfg.attachSmoothing = m_params->attachSmoothing.load();
    mcfg.attachMaxStepPx = m_params->attachMaxStepPx.load();

    // HID
    mcfg.useFaHid = m_params->mouse_useFaHid.load(std::memory_order_relaxed);
    mcfg.fa.vid = (uint16_t)m_params->mouse_fa_vid.load(std::memory_order_relaxed);
    mcfg.fa.pid = (uint16_t)m_params->mouse_fa_pid.load(std::memory_order_relaxed);
    mcfg.fa.frame_head = (uint8_t)m_params->mouse_fa_head.load(std::memory_order_relaxed);
    mcfg.fa.frame_tail = (uint8_t)m_params->mouse_fa_tail.load(std::memory_order_relaxed);
    mcfg.fa.feature_report_id = (uint8_t)m_params->mouse_fa_report_id.load(std::memory_order_relaxed);

    mcfg.faSendEnableDisable = m_params->mouse_fa_sendEnable.load(std::memory_order_relaxed);
    mcfg.faReconnectMs = m_params->mouse_fa_reconnectMs.load(std::memory_order_relaxed);
    mcfg.faActive = false;

    if (!m_mouse->initialize(mcfg)) {
        // ✅ 这里给你一个“更实用”的选择：如果 FaHID 失败，不要让程序整体启动失败
        // 如果你希望“没有 HID 也能跑（只是鼠标不动）”，就改成：返回 true 并提示断开
        // return Fail(L"MouseController::initialize", L"mouse init failed: " + std::wstring(m_mouse->lastError().begin(), m_mouse->lastError().end()));

        return Fail(L"MouseController::initialize",
            L"mouse init failed. (常见：FaHID 设备不存在/VIDPID不对/驱动未装)\n"
            L"建议：先把 mouse_useFaHid 设为 false 验证其它链路是否正常。");
    }

    // 7) tracking reset
    m_tracker = EyeTrackerState{};
    m_lock = TargetTrack{};

    OutputDebugStringW(L"[initResources OK]\n");
    return true;
}


void BackendWorker::runLoop() {
    const int ROI = 416;

    using clock = std::chrono::steady_clock;

    auto nextInfer = clock::now();
    constexpr float MOVE_TICK_MS = 5.0f;
    auto nextMove = clock::now();

    bool lastPgUp = false; // PageUp edge trigger
    bool lastF6 = false;  // F6 cycles FPS cap steps


    while (m_running.load()) {
        // PageUp: 项目总开关（边沿触发）
        const bool nowPgUp = (GetAsyncKeyState(VK_PRIOR) & 0x8000) != 0;
        if (nowPgUp && !lastPgUp) {
            const bool newEnabled = !m_params->enabled.load(std::memory_order_relaxed);
            setEnabled(newEnabled);
        }
        lastPgUp = nowPgUp;

        // 60/90/120/144 循环切换（按一次F6只触发一次）
        if (GetAsyncKeyState(VK_F6) & 1) {   // ✅ &1：自带“从上次调用以来是否按下过”
            int cur = m_params->fps.load(std::memory_order_relaxed);
            cur = SanitizeFpsCap(cur);
            int nxt = NextFpsCap(cur);
            m_params->fps.store(nxt, std::memory_order_relaxed);

            // 可选：放一个提示音，确认确实触发到了（调试用）
            MessageBeep(MB_OK);
        }


        if (m_params->reqRestart.exchange(false)) {
            m_tracker = EyeTrackerState{};
            m_lock = TargetTrack{};
            if (m_mouse) {
                m_mouse->resetAttach();
                m_mouse->planReset();
            }
        }
        if (m_params->reqClear.exchange(false)) {
            m_tracker = EyeTrackerState{};
            m_lock = TargetTrack{};
            if (m_mouse) {
                m_mouse->resetAttach();
                m_mouse->planReset();
            }
        }

        if (!m_params->enabled.load()) {
            if (m_mouse) {
                MouseControllerConfig cfg = m_mouse->config();
                cfg.faActive = false;
                m_mouse->updateConfig(cfg);
                m_mouse->resetAttach();
                m_mouse->planReset();
            }
            nextInfer = clock::now();
            nextMove = clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        const bool nowDown = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;

        // ✅ 右键按下：开始一次新的“强边界”锁定会话 => 清状态（干净）
        if (nowDown && !m_wasDown) {
            if (m_mouse) {
                m_mouse->resetAttach();
                m_mouse->planReset();
            }
            m_tracker = EyeTrackerState{};
            m_lock = TargetTrack{};
        }

        // ✅ 右键松开：彻底停止 + 清状态（干净）
        if (!nowDown && m_wasDown) {
            if (m_mouse) {
                m_mouse->resetAttach();
                m_mouse->planReset();
            }
            m_tracker = EyeTrackerState{};
            m_lock = TargetTrack{};
        }

        m_wasDown = nowDown;
        const bool attach = nowDown;

        // --- UI stats publish (cheap) ---
        auto publishUi = [&]() {
            if (!m_params) return;
            BackendWorker::UiStats us;
            {
                std::lock_guard<std::mutex> lk(m_uiMu);
                // keep current fps values; copy then edit
                us = m_uiStats;
                // lock/tracking
                us.track_count = (int)m_tracker.tracks.size();
                us.locked_uid = m_tracker.lockedUid;
                us.lock_has = m_lock.has;
                us.lock_residual = m_lock.residual;
                us.lock_uid = m_lock.uid;
                us.lock_miss = m_lock.miss;
                us.lock_srank = m_lock.sRank;
                // capture stats (from DxgiDup)
                if (m_dup) {
                    const HRESULT hr = m_dup->last_acquire_hr();
                    us.cap_last_hr = (long)hr;
                    us.cap_fail_streak = m_dup->cap_fail_streak();
                    us.cap_recover_count = m_dup->cap_recover_count();
                    us.cap_black_streak = m_dup->cap_black_streak();
                    us.cap_permission_hint = (hr == E_ACCESSDENIED || hr == (HRESULT)0x887A002E /*DXGI_ERROR_ACCESS_DENIED*/);
                }
                // mouse stats
                if (m_mouse) {
                    int dx = 0, dy = 0;
                    m_mouse->getLastMove(dx, dy);
                    us.dx = dx; us.dy = dy;
                    us.fa_connected = m_mouse->faConnected();
                    us.fa_active = m_mouse->faActive();
                }
                m_uiStats = us;
            }
            };


        int fpsCap = SanitizeFpsCap(m_params->fps.load(std::memory_order_relaxed));
        m_params->fps.store(fpsCap, std::memory_order_relaxed);

        
        const auto inferInterval = std::chrono::microseconds(
            (int)std::llround(1'000'000.0 / (double)fpsCap)
        );




        // ✅ 每圈都同步鼠标配置（包含硬件 enable/disable 边沿 + 动态磁铁参数）
        if (m_mouse) {
            MouseControllerConfig cfg = m_mouse->config();

            // stop / safety clamp
            cfg.attachDeadzonePx = m_params->attachDeadzonePx.load(std::memory_order_relaxed);

            cfg.attachGain = m_params->attachGain.load(std::memory_order_relaxed);
            cfg.attachSmoothing = m_params->attachSmoothing.load(std::memory_order_relaxed);
            cfg.attachMaxStepPx = m_params->attachMaxStepPx.load(std::memory_order_relaxed);

            // HID
            cfg.useFaHid = m_params->mouse_useFaHid.load(std::memory_order_relaxed);
            cfg.fa.vid = (uint16_t)m_params->mouse_fa_vid.load(std::memory_order_relaxed);
            cfg.fa.pid = (uint16_t)m_params->mouse_fa_pid.load(std::memory_order_relaxed);
            cfg.fa.frame_head = (uint8_t)m_params->mouse_fa_head.load(std::memory_order_relaxed);
            cfg.fa.frame_tail = (uint8_t)m_params->mouse_fa_tail.load(std::memory_order_relaxed);
            cfg.fa.feature_report_id = (uint8_t)m_params->mouse_fa_report_id.load(std::memory_order_relaxed);

            cfg.faSendEnableDisable = m_params->mouse_fa_sendEnable.load(std::memory_order_relaxed);
            cfg.faReconnectMs = m_params->mouse_fa_reconnectMs.load(std::memory_order_relaxed);

            cfg.faActive = (m_params->enabled.load(std::memory_order_relaxed) && attach);

            m_mouse->updateConfig(cfg);
        }

        auto now = clock::now();

        if (now >= nextInfer) {
            nextInfer = now + inferInterval;
            processFrame(ROI, attach);
        }

        if (now >= nextMove) {
            nextMove = now + std::chrono::milliseconds((int)MOVE_TICK_MS);

            // ✅ 关键：残影期绝不 planTick（不移动）
            if (m_mouse && m_params->enabled.load() && attach && m_lock.has && !m_lock.residual) {
                m_mouse->planTick();
            }
            else if (m_mouse) {
                m_mouse->planReset();
            }
        }

        publishUi();

        auto wake = std::min(nextInfer, nextMove);
        now = clock::now();
        if (wake > now) {
            auto remain = wake - now;

            // 余量大先睡（留 1ms 安全边界防 oversleep）
            if (remain > std::chrono::milliseconds(2)) {
                std::this_thread::sleep_for(remain - std::chrono::milliseconds(1));
            }

            // 余量小用短自旋补齐（保证 90/120/144 这种小周期更准）
            while (clock::now() < wake) {
                Sleep(0); // 或者 _mm_pause(); 但Sleep(0)更省事
            }
        }

    }
}

void BackendWorker::processFrame(int roi, bool attach) {
    RoiInfo info{};
    ID3D11Texture2D* roiTexRaw = m_dup->acquire_roi_tex(roi, info, 0); // non-blocking; reuse on timeout
    if (!roiTexRaw) return;

    // --- capture fps accounting (count only when a NEW desktop frame arrived) ---
    if (m_dup && m_dup->last_acquire_hr() == S_OK) {
        using clock = std::chrono::steady_clock;

        auto countCapFpsIfNewFrame = [&]() {
            if (!m_dup) return;
            if (m_dup->last_acquire_hr() != S_OK) return;  // ✅ 只有新帧才计数

            const auto now = clock::now();
            if (m_capFrames == 0) m_capT0 = now;
            m_capFrames++;

            const double dt = std::chrono::duration<double>(now - m_capT0).count();
            if (dt >= 1.0) {
                const double fps = (double)m_capFrames / dt;
                m_capFrames = 0;
                m_capT0 = now;
                std::lock_guard<std::mutex> lk(m_uiMu);
                m_uiStats.cap_fps = fps;
            }
            };
    }


    ComPtr<ID3D11Texture2D> roiTex;
    roiTex.Attach(roiTexRaw);

    auto stream = (cudaStream_t)m_stream;
    auto cuRes = (cudaGraphicsResource*)m_cuRes;

    ID3D11Texture2D* lastTex = (ID3D11Texture2D*)m_regTex;
    if (!lastTex || lastTex != roiTex.Get()) {
        if (cuRes) {
            cudaGraphicsUnregisterResource(cuRes);
            cuRes = nullptr;
        }
        if (!CheckCuda(cudaGraphicsD3D11RegisterResource(&cuRes, roiTex.Get(), cudaGraphicsRegisterFlagsNone),
            "cudaGraphicsD3D11RegisterResource")) {
            return;
        }
        m_cuRes = (void*)cuRes;
        m_regTex = (void*)roiTex.Get();
    }

    if (!CheckCuda(cudaGraphicsMapResources(1, &cuRes, stream), "cudaGraphicsMapResources"))
        return;

    cudaArray_t arr = nullptr;
    if (!CheckCuda(cudaGraphicsSubResourceGetMappedArray(&arr, cuRes, 0, 0),
        "cudaGraphicsSubResourceGetMappedArray")) {
        cudaGraphicsUnmapResources(1, &cuRes, stream);
        return;
    }

    preprocess_bgra8_to_f32_nchw(arr, roi, (float*)m_trt->d_input(), stream);
    cudaError_t pe = cudaPeekAtLastError();
    if (pe != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &cuRes, stream);
        return;
    }

    if (!CheckCuda(cudaGraphicsUnmapResources(1, &cuRes, stream), "cudaGraphicsUnmapResources"))
        return;

    if (!m_trt->infer(stream)) return;

    // --- infer fps accounting ---
    {
        using clock = std::chrono::steady_clock;
        const auto now = clock::now();
        if (m_inferFrames == 0) m_inferT0 = now;
        m_inferFrames++;
        const double dt = std::chrono::duration<double>(now - m_inferT0).count();
        if (dt >= 1.0) {
            const double fps = (double)m_inferFrames / dt;
            m_inferFrames = 0;
            m_inferT0 = now;
            std::lock_guard<std::mutex> lk(m_uiMu);
            m_uiStats.infer_fps =fps;
        }
    }


    // ---------- 1) 消费一个已完成 slot（不阻塞） ----------
    int readyIdx = -1;
    uint64_t bestSeq = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < 2; ++i) {
        if (!m_outSlots[i].pending) continue;

        cudaError_t q = cudaEventQuery((cudaEvent_t)m_outSlots[i].ev);
        if (q == cudaSuccess) {
            if (m_outSlots[i].seq < bestSeq) {
                bestSeq = m_outSlots[i].seq;
                readyIdx = i;
            }
        }
        else if (q == cudaErrorNotReady) {
            // 还没好
        }
        else {
            m_outSlots[i].pending = false;
        }
    }

    if (readyIdx >= 0) {
        auto& slot = m_outSlots[readyIdx];

        if (attach && slot.attach && m_params->enabled.load()) {
            RoiInfo snap{};
            snap.ox = slot.ox; snap.oy = slot.oy;
            snap.full_w = slot.full_w; snap.full_h = slot.full_h;

            float max_score = 0.0f;

            auto dets = decode_yolo_1x8xN_f32(
                slot.host,
                slot.roi,
                slot.conf_thres,
                slot.iou_thres,
                snap.ox, snap.oy, snap.full_w, snap.full_h,
                &max_score);

            dets.erase(
                std::remove_if(dets.begin(), dets.end(),
                    [](const Det& d) { return d.cls != 0; }),
                dets.end()
            );

            // ✅ 多目标轨迹池 + 输出锁定目标（可能 residual）
                        // UI stats: det_count
            {
                std::lock_guard<std::mutex> lk(m_uiMu);
                m_uiStats.det_count = (int)dets.size();
            }

            UpdateTracking(dets, snap, m_params, m_tracker, m_lock);

            // UI stats: lock/tracks snapshot (updated on new results)
            {
                std::lock_guard<std::mutex> lk(m_uiMu);
                m_uiStats.track_count = (int)m_tracker.tracks.size();
                m_uiStats.locked_uid = m_tracker.lockedUid;
                m_uiStats.lock_has = m_lock.has;
                m_uiStats.lock_residual = m_lock.residual;
                m_uiStats.lock_uid = m_lock.uid;
                m_uiStats.lock_miss = m_lock.miss;
                m_uiStats.lock_srank = m_lock.sRank;
            }


            if (m_mouse) {
                const bool enabled = m_params->enabled.load();
                const bool hasRealTarget = (m_lock.has && !m_lock.residual);

                if (enabled && attach && hasRealTarget) {
                    int currentFps = std::max(1, m_params->fps.load());
                    float framePeriodMs = 1000.0f / (float)currentFps;
                    safeAttach(*m_mouse, m_lock, framePeriodMs);
                }
                else if (enabled && attach && m_lock.has && m_lock.residual) {
                    // ✅ 残影期：不定位、不移动（干净）
                    m_mouse->planReset();
                }
                else {
                    m_mouse->resetAttach();
                    m_mouse->planReset();
                }
            }
        }

        slot.pending = false;
    }

    // ---------- 2) 本帧输出 async 拷到空闲 slot ----------
    if (!attach) return;

    int freeIdx = -1;
    for (int i = 0; i < 2; ++i) {
        if (!m_outSlots[i].pending) { freeIdx = i; break; }
    }
    if (freeIdx < 0) return;

    auto& w = m_outSlots[freeIdx];

    w.roi = roi;
    w.ox = info.ox; w.oy = info.oy; w.full_w = info.full_w; w.full_h = info.full_h;
    w.attach = attach;
    w.conf_thres = m_params->conf_thres.load();
    w.iou_thres = m_params->iou_thres.load();
    w.seq = ++m_outSeq;

    if (!CheckCuda(cudaMemcpyAsync(w.host, m_trt->d_output(), m_outBytes, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(DtoH slot)")) {
        w.pending = false;
        return;
    }
    if (!CheckCuda(cudaEventRecord((cudaEvent_t)w.ev, stream), "cudaEventRecord")) {
        w.pending = false;
        return;
    }

    w.pending = true;
}

bool BackendWorker::safeAttach(MouseController& mouse, TargetTrack& tr, float framePeriodMs) {
    const int sw = GetSystemMetrics(SM_CXSCREEN);
    const int sh = GetSystemMetrics(SM_CYSCREEN);

    // ✅ 不要在这里改写 tr（锁定状态应该只由 tracking 决定）
    if (!std::isfinite(tr.cx) || !std::isfinite(tr.cy) || sw <= 1 || sh <= 1) {
        mouse.resetAttach();
        return false;
    }

    long lx = std::lround(tr.cx);
    long ly = std::lround(tr.cy);

    lx = std::clamp<long>(lx, 0, sw - 1);
    ly = std::clamp<long>(ly, 0, sh - 1);

    constexpr float MOVE_TICK_MS = 5.0f;
    constexpr float ALPHA = 0.7f;

    mouse.planNewTarget((int)lx, (int)ly, framePeriodMs, MOVE_TICK_MS, ALPHA);
    return true;
}

void BackendWorker::cleanup() {
    // 先释放双缓冲
    for (int i = 0; i < 2; ++i) {
        if (m_outSlots[i].ev) {
            cudaEventDestroy((cudaEvent_t)m_outSlots[i].ev);
            m_outSlots[i].ev = nullptr;
        }
        if (m_outSlots[i].host) {
            cudaFreeHost(m_outSlots[i].host);
            m_outSlots[i].host = nullptr;
        }
        m_outSlots[i].pending = false;
        m_outSlots[i].seq = 0;
    }
    m_outBytes = 0;
    m_outSeq = 0;

    auto stream = (cudaStream_t)m_stream;
    auto cuRes = (cudaGraphicsResource*)m_cuRes;

    if (cuRes) {
        cudaGraphicsUnregisterResource(cuRes);
        m_cuRes = nullptr;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        m_stream = nullptr;
    }

    if (m_mouse) {
        m_mouse->shutdown();
        m_mouse.reset();
    }
    if (m_trt) {
        m_trt->shutdown();
        m_trt.reset();
    }
    if (m_dup) {
        m_dup->shutdown();
        m_dup.reset();
    }

    m_regTex = nullptr;

    // ✅ 清理多目标轨迹池与锁定（干净）
    m_tracker = EyeTrackerState{};
    m_lock = TargetTrack{};
}
