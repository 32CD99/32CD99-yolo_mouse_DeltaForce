#include "tracking.h"
#include "runtime_params.h"
#include "dxgi_dup.h"
#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

static inline float dist2(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}

// 尺寸相近：面积比 + 长宽比（支持 sizeRatio=1.0 极严）
static inline bool size_similar(float w, float h, float refW, float refH, float sizeRatio) {
    const float eps = 1e-6f;
    if (!(w > eps && h > eps && refW > eps && refH > eps)) return true;

    float s = std::max(1.0f, sizeRatio);
    const float tol = 1e-3f;
    float lo = (1.0f / s) - tol;
    float hi = s + tol;

    float area = w * h;
    float refA = refW * refH;
    if (!(area > eps && refA > eps)) return true;

    float rA = area / refA;
    if (rA < lo || rA > hi) return false;

    float rAR = (w / h) / (refW / refH);
    if (rAR < lo || rAR > hi) return false;

    return true;
}

static inline void push_hist(EyeTrack& t, int win, uint8_t real) {
    if (win <= 0) return;
    if ((int)t.hist.size() >= win) {
        t.realCount -= (int)t.hist.front();
        t.hist.pop_front();
    }
    t.hist.push_back(real);
    t.realCount += (int)real;
}

// 稳定判定：最近 win 帧窗口内，“真实匹配帧数” >= needFrames 才算稳定
// 允许 seen>=needFrames 就提前判定（不用等满 win）
static inline bool is_stable_need(const EyeTrack& t, int win, int needFrames) {
    if (win <= 0) return false;
    int need = std::clamp(needFrames, 1, win);
    const int seen = (int)t.hist.size();
    return (seen >= need && t.realCount >= need);
}
static inline int find_track_idx_by_uid(const EyeTrackerState& st, int uid) {
    if (uid == 0) return -1;
    for (int i = 0; i < (int)st.tracks.size(); ++i) {
        if (st.tracks[i].uid == uid) return i;
    }
    return -1;
}

void UpdateTracking(const std::vector<Det>& dets,
    const RoiInfo& info,
    RuntimeParams* params,
    EyeTrackerState& st,
    TargetTrack& lockOut)
{
    lockOut = TargetTrack{};

    if (!params) return;

    if (!params->enabled.load()) {
        st = EyeTrackerState{};
        return;
    }

    const float screenCx = info.full_w * 0.5f;
    const float screenCy = info.full_h * 0.5f;

    const float maxReassocPx = params->max_reassoc_px.load();
    const float maxReassocPx2 = maxReassocPx * maxReassocPx;

    // ✅ 一个参数 max_miss：TTL + 连续性窗口
    const int   maxMiss = std::max(1, params->max_miss.load());
    const float sizeRatio = std::max(1.0f, params->residual_size_ratio.load());

    const int   stableNeed = std::clamp(params->stable_ratio.load(), 1, maxMiss);

    const float AIM_UP_RATIO = params->aim_up_ratio.load();
    auto aimOf = [&](const Det& b, float& ax, float& ay, float& bw, float& bh) {
        bw = (b.x2 - b.x1);
        bh = (b.y2 - b.y1);
        ax = (b.x1 + b.x2) * 0.5f;
        ay = (b.y1 + b.y2) * 0.5f - bh * AIM_UP_RATIO;
        };

    // ---- 提取 det 特征
    struct DFeat { float ax, ay, w, h; };
    std::vector<DFeat> df;
    df.reserve(dets.size());
    for (const auto& d : dets) {
        float ax, ay, bw, bh;
        aimOf(d, ax, ay, bw, bh);
        if (!std::isfinite(ax) || !std::isfinite(ay)) continue;
        df.push_back({ ax, ay, bw, bh });
    }

    const int D = (int)df.size();
    const int T = (int)st.tracks.size();

    // ---- 关联：全pair按距离排序贪心匹配（带距离+尺寸门控）
    std::vector<int> trackToDet(T, -1);
    std::vector<int> detToTrack(D, -1);

    struct Pair { int ti; int di; float c; };
    std::vector<Pair> pairs;
    pairs.reserve((size_t)T * (size_t)D);

    for (int ti = 0; ti < T; ++ti) {
        const auto& tr = st.tracks[ti];
        for (int di = 0; di < D; ++di) {
            const auto& f = df[di];

            float d2 = dist2(f.ax, f.ay, tr.cx, tr.cy);
            if (d2 > maxReassocPx2) continue;
            if (!size_similar(f.w, f.h, tr.w, tr.h, sizeRatio)) continue;

            pairs.push_back({ ti, di, d2 });
        }
    }

    std::sort(pairs.begin(), pairs.end(),
        [](const Pair& a, const Pair& b) { return a.c < b.c; });

    for (const auto& p : pairs) {
        if (trackToDet[p.ti] != -1) continue;
        if (detToTrack[p.di] != -1) continue;
        trackToDet[p.ti] = p.di;
        detToTrack[p.di] = p.ti;
    }

    // ---- 更新已有轨迹：最新真实框优先；没框就残影（只用于重联，不用于定位）
    for (int ti = 0; ti < T; ++ti) {
        auto& tr = st.tracks[ti];
        int di = trackToDet[ti];

        if (di >= 0) {
            // ✅ 最新框优先：只要匹配到真实框，就立刻更新
            tr.cx = df[di].ax;
            tr.cy = df[di].ay;
            tr.w = df[di].w;
            tr.h = df[di].h;

            tr.miss = 0;
            tr.matched = true;
            push_hist(tr, maxMiss, 1);
        }
        else {
            // 残影：位置不变，只累计 miss（用于身份延续）
            tr.miss++;
            tr.matched = false;
            push_hist(tr, maxMiss, 0);
        }

        tr.stable = is_stable_need(tr, maxMiss, stableNeed);
    }

    // ---- 新建未匹配 det 的轨迹
    for (int di = 0; di < D; ++di) {
        if (detToTrack[di] != -1) continue;

        EyeTrack nt{};
        nt.uid = st.nextUid++;
        nt.cx = df[di].ax; nt.cy = df[di].ay;
        nt.w = df[di].w;  nt.h = df[di].h;

        nt.miss = 0;
        nt.matched = true;
        nt.hist.clear();
        nt.realCount = 0;
        push_hist(nt, maxMiss, 1);
        nt.stable = is_stable_need(nt, maxMiss, stableNeed);

        st.tracks.push_back(std::move(nt));
    }

    // ---- 如果锁定轨迹已经死亡：彻底断开（清空轨迹池 + 清锁）
    bool lockLost = false;
    if (st.lockedUid != 0) {
        int li = find_track_idx_by_uid(st, st.lockedUid);
        if (li < 0 || st.tracks[li].miss >= maxMiss) {
            lockLost = true;
        }
    }

    // 先清理死亡轨迹
    st.tracks.erase(
        std::remove_if(st.tracks.begin(), st.tracks.end(),
            [&](const EyeTrack& tr) { return tr.miss >= maxMiss; }),
        st.tracks.end()
    );

    if (lockLost) {
        // ✅ 干干净净：不让别的轨迹“接盘”，直接全清
        st = EyeTrackerState{};
        lockOut = TargetTrack{};
        return;
    }

    // ---- 计算稳定轨迹的 sRank（仅用于显示/调试）
    struct Cand { int idx; float d2; };
    std::vector<Cand> cands;
    cands.reserve(st.tracks.size());

    for (int i = 0; i < (int)st.tracks.size(); ++i) {
        if (!st.tracks[i].stable) continue;
        float d2 = dist2(st.tracks[i].cx, st.tracks[i].cy, screenCx, screenCy);
        cands.push_back({ i, d2 });
    }

    std::sort(cands.begin(), cands.end(),
        [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });

    // ---- 还没锁：只从“稳定且本帧有真实框(matched)”里选一个锁（强边界）
    if (st.lockedUid == 0) {
        for (const auto& c : cands) {
            if (st.tracks[c.idx].matched) {
                st.lockedUid = st.tracks[c.idx].uid;
                break;
            }
        }
    }

    // ---- 输出锁定目标：残影期 has=true 但 residual=true；且 cx/cy 不用于定位
    if (st.lockedUid != 0) {
        int li = find_track_idx_by_uid(st, st.lockedUid);
        if (li >= 0) {
            const auto& lk = st.tracks[li];

            lockOut.has = true;
            lockOut.uid = lk.uid;
            lockOut.miss = lk.miss;

            // sRank
            lockOut.sRank = -1;
            for (int r = 0; r < (int)cands.size(); ++r) {
                if (cands[r].idx == li) { lockOut.sRank = r; break; }
            }

            // ✅ 关键：残影不定位
            lockOut.residual = !lk.matched;
            if (lk.matched) {
                lockOut.cx = lk.cx;
                lockOut.cy = lk.cy;
                lockOut.w = lk.w;
                lockOut.h = lk.h;
            }
            else {
                // 残影期间不提供有效目标点（保持为0即可），用于强制上层不移动
                lockOut.cx = 0.f; lockOut.cy = 0.f;
                lockOut.w = 0.f; lockOut.h = 0.f;
            }
        }
        else {
            st.lockedUid = 0;
        }
    }
}

