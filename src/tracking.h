#pragma once
#include <vector>
#include <deque>
#include <cstdint>

struct RuntimeParams;
struct RoiInfo;
struct Det;

// 给鼠标用的锁定目标输出
struct TargetTrack {
    bool  has = false;        // 是否存在“锁定对象”（锁还活着）
    bool  residual = false;   // true=本帧没真实框（残影期）=> 绝不能用于定位
    int   uid = 0;            // 锁定轨迹ID
    int   miss = 0;           // 连续没真实框的帧数（TTL计数）
    int   sRank = -1;         // 稳定轨迹按中心距离排序的名次（0=s0），但锁定不一定是0

    float cx = 0.f, cy = 0.f; // 只有 residual=false 时，才代表“最新真实框”的目标点
    float w = 0.f, h = 0.f;
};

struct EyeTrack {
    int   uid = 0;

    float cx = 0.f, cy = 0.f;
    float w = 0.f, h = 0.f;

    int   miss = 0;          // 连续缺失真实检测的帧数
    bool  matched = false;   // 本帧是否匹配到真实框

    // 最近 max_miss 帧：1=真实框，0=残影
    std::deque<uint8_t> hist;
    int   realCount = 0;

    bool  stable = false;    // 稳定判定（窗口内真实匹配帧数达到阈值）
};

struct EyeTrackerState {
    std::vector<EyeTrack> tracks;
    int nextUid = 1;

    // ✅ 强锁定：鼠标永远跟这条轨迹，除非它死亡
    int lockedUid = 0;
};

void UpdateTracking(const std::vector<Det>& dets,
    const RoiInfo& info,
    RuntimeParams* params,
    EyeTrackerState& st,
    TargetTrack& lockOut);
