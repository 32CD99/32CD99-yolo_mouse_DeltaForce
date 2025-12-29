#pragma once
#include <atomic>

struct RuntimeParams {
    // ---------------- 检测阈值 ----------------
    std::atomic<float> conf_thres{ 0.40f };
    std::atomic<float> iou_thres{ 0.10f };

    // ---------------- Tracking 追踪 ----------------
    std::atomic<int>   max_miss{ 10 };                 // 允许丢失帧数（超过就丢弃 track）
    std::atomic<float> max_reassoc_px{ 60.0f };        // det->track 最大重联距离（px）
    std::atomic<float> aim_up_ratio{ 0.15f };          // 瞄点上移比例：aim_y = cy - h * ratio
    std::atomic<float> residual_size_ratio{ 1.6f };    // 面积相似允许比例（用于重联门控）
    std::atomic<int>   stable_ratio{ 5 };              // 稳定判定：最近窗口内"真实匹配"至少多少帧才算稳定（与 max_miss 解耦）

    // mouse attach (near)
    std::atomic<int>   attachDeadzonePx{ 1 };
    std::atomic<float> attachGain{ 1.6f };
    std::atomic<int>   attachMaxStepPx{ 30 };
    std::atomic<float> attachSmoothing{ 1.0f };

    // ---------------- STM32 HID Feature (FaController) ----------------
    std::atomic<bool>  mouse_useFaHid{ true };
    std::atomic<int>   mouse_fa_vid{ 0x0483 };
    std::atomic<int>   mouse_fa_pid{ 0x5750 };
    std::atomic<int>   mouse_fa_head{ 0xAA };
    std::atomic<int>   mouse_fa_tail{ 0x55 };
    std::atomic<int>   mouse_fa_report_id{ 0x02 };
    std::atomic<bool>  mouse_fa_sendEnable{ true };
    std::atomic<int>   mouse_fa_reconnectMs{ 1000 };

    // ---------------- UI/backend control ----------------
    std::atomic<bool> enabled{ true };
    std::atomic<bool> reqRestart{ false };
    std::atomic<bool> reqClear{ false };
    std::atomic<int>  fps{ 80 };
};
