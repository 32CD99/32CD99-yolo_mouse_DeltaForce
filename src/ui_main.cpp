#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include <cwchar>
#include <string>
#include <algorithm>

#include "ui_main.h"
#include "runtime_params.h"
#include "backend_worker.h"

// 如果你的 CMake 没定义 UNICODE/_UNICODE，也能编译；但建议在 CMake 里统一打开 UNICODE。
// 这里统一使用 W 系列 API。
#pragma comment(lib, "comctl32.lib")

namespace {

    // CreateWindowExW 只能传一个 lpParam，这里把 params + worker 打包
    struct UiCreateArgs {
        RuntimeParams* params = nullptr;
        BackendWorker* worker = nullptr;
    };

    constexpr UINT_PTR kUiTimerId = 1;
    constexpr UINT     kUiTimerMs = 200;
    constexpr UINT     WM_APP_APPLY_EDIT = WM_APP + 10;

    constexpr int kMargin = 12;
    constexpr int kGap = 12;
    constexpr int kRowH = 24;
    constexpr int kLblW = 130;
    constexpr int kEditW = 120;

    enum CtrlId : int {
        // 顶部
        ID_CHK_TOPMOST = 100,

        // 调试区 value labels
        ID_LBL_CAP_FPS_VAL = 200,
        ID_LBL_INF_FPS_VAL = 201,
        ID_LBL_DET_CNT_VAL = 202,
        ID_LBL_TRACK_VAL = 203,
        ID_LBL_DXDY_VAL = 204,
        ID_LBL_FAHID_CONN_VAL = 205,
        ID_LBL_FAHID_ACT_VAL = 206,
        ID_LBL_CAPTURE_ERR_VAL = 207,

        // 状态区（只读勾选展示）
        ID_CHK_MASTER_STATUS = 300,
        ID_CHK_FAHID_STATUS = 301,

        // Fa(HID) 参数编辑
        ID_EDIT_FA_VID = 400,
        ID_EDIT_FA_PID = 401,
        ID_EDIT_FA_HEAD = 402,
        ID_EDIT_FA_TAIL = 403,
        ID_EDIT_FA_RID = 404,
        ID_EDIT_FA_RECONN = 405,

        // 检测参数
        ID_EDIT_CONF_THRES = 500,
        ID_EDIT_IOU_THRES = 501,

        // Tracking
        ID_EDIT_MAX_MISS = 600,
        ID_EDIT_MAX_REASSOC_PX = 601,
        ID_EDIT_AIM_UP_RATIO = 602,
        ID_EDIT_RESIDUAL_SIZE_RATIO = 603,
        ID_EDIT_STABLE_RATIO = 604,

        // 鼠标控制
        ID_EDIT_ATTACH_DEADZONE = 700,
        ID_EDIT_ATTACH_GAIN = 701,
        ID_EDIT_ATTACH_MAXSTEP = 702,
        ID_EDIT_ATTACH_SMOOTHING = 703,
    };

    struct UiState {
        RuntimeParams* params = nullptr;
        BackendWorker* worker = nullptr;

        HFONT font = nullptr;

        // panels
        RECT rcDbg{};
        RECT rcStatus{};
        RECT rcFa{};
        RECT rcDetect{};
        RECT rcTrack{};
        RECT rcMouse{};

        // top
        HWND chkTopmost = nullptr;

        // debug labels (value)
        HWND vCapFps = nullptr;
        HWND vInfFps = nullptr;
        HWND vDetCnt = nullptr;
        HWND vTrack = nullptr;
        HWND vDxDy = nullptr;
        HWND vFaConn = nullptr;
        HWND vFaAct = nullptr;
        HWND vCapErr = nullptr;

        // status checkboxes (readonly)
        HWND chkMaster = nullptr;
        HWND chkFaHid = nullptr;

        // edits
        HWND eFaVid = nullptr;
        HWND eFaPid = nullptr;
        HWND eFaHead = nullptr;
        HWND eFaTail = nullptr;
        HWND eFaRid = nullptr;
        HWND eFaReconn = nullptr;

        HWND eConf = nullptr;
        HWND eIou = nullptr;

        HWND eMaxMiss = nullptr;
        HWND eMaxReassoc = nullptr;
        HWND eAimUp = nullptr;
        HWND eSizeRatio = nullptr;
        HWND eStableNeed = nullptr;

        HWND eAttachDeadzone = nullptr;
        HWND eAttachGain = nullptr;
        HWND eAttachMaxStep = nullptr;
        HWND eAttachSmooth = nullptr;
        bool topmost = false;
    };

    static UiState* GetState(HWND hwnd) {
        return reinterpret_cast<UiState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }

    static void SetCtlFont(HWND h, HFONT f) {
        if (h && f) SendMessageW(h, WM_SETFONT, (WPARAM)f, TRUE);
    }

    static HFONT CreateUiFont(HWND hwnd) {
        HDC hdc = GetDC(hwnd);
        int dpi = GetDeviceCaps(hdc, LOGPIXELSY);
        ReleaseDC(hwnd, hdc);
        int height = -MulDiv(9, dpi, 72);
        return CreateFontW(height, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
    }

    static HWND MakeLabel(HWND parent, int id, const wchar_t* text) {
        return CreateWindowExW(0, L"STATIC", text,
            WS_CHILD | WS_VISIBLE,
            0, 0, 10, 10, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    }

    static HWND MakeValue(HWND parent, int id) {
        // 右侧 value，用 STATIC
        return CreateWindowExW(0, L"STATIC", L"\u2014",
            WS_CHILD | WS_VISIBLE,
            0, 0, 10, 10, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    }

    static HWND MakeEdit(HWND parent, int id) {
        return CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
            WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
            0, 0, 10, 10, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    }

    static HWND MakeCheck(HWND parent, int id, const wchar_t* text) {
        return CreateWindowExW(0, L"BUTTON", text,
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
            0, 0, 10, 10, parent, (HMENU)(INT_PTR)id, nullptr, nullptr);
    }

    static void SetReadOnlyCheck(HWND hChk) {
        // 用 disable 达到“只读展示”效果（仍能显示勾选状态）
        if (hChk) EnableWindow(hChk, FALSE);
    }

    static void SetEditTextF(HWND hEdit, float v, int decimals = 3) {
        wchar_t buf[64]{};
        if (decimals < 0) decimals = 3;
        wchar_t fmt[16]{};
        swprintf_s(fmt, L"%%.%df", decimals);
        swprintf_s(buf, fmt, v);
        SetWindowTextW(hEdit, buf);
    }

    static void SetEditTextIHex(HWND hEdit, int v) {
        wchar_t buf[64]{};
        swprintf_s(buf, L"0x%X", v);
        SetWindowTextW(hEdit, buf);
    }

    static void SetEditTextI(HWND hEdit, int v) {
        wchar_t buf[64]{};
        swprintf_s(buf, L"%d", v);
        SetWindowTextW(hEdit, buf);
    }

    static bool TryParseFloat(HWND hEdit, float* out) {
        wchar_t buf[128]{};
        GetWindowTextW(hEdit, buf, 127);
        wchar_t* end = nullptr;
        double v = wcstod(buf, &end);
        if (end == buf) return false;
        *out = (float)v;
        return true;
    }

    static bool TryParseIntDecOrHex(HWND hEdit, int* out) {
        wchar_t buf[128]{};
        GetWindowTextW(hEdit, buf, 127);
        // 支持：123 / 0x1A / 1A
        const wchar_t* s = buf;
        while (*s == L' ' || *s == L'\t') ++s;
        int base = 10;
        if (s[0] == L'0' && (s[1] == L'x' || s[1] == L'X')) base = 16;
        wchar_t* end = nullptr;
        long v = wcstol(s, &end, base);
        if (end == s) {
            // 退一步：如果用户直接输入 "1A"（无 0x），按 16 解析
            v = wcstol(s, &end, 16);
            if (end == s) return false;
        }
        *out = (int)v;
        return true;
    }

    // 修正后的 ApplyEditById 函数 - 替换原来的函数
    static void ApplyEditById(HWND hwnd, int id) {
        UiState* st = GetState(hwnd);
        if (!st || !st->params) return;

        RuntimeParams& p = *st->params;

        switch (id) {
        case ID_EDIT_CONF_THRES: {
            float v;
            if (TryParseFloat(GetDlgItem(hwnd, ID_EDIT_CONF_THRES), &v)) p.conf_thres.store(std::clamp(v, 0.0f, 1.0f));
            SetEditTextF(GetDlgItem(hwnd, ID_EDIT_CONF_THRES), p.conf_thres.load(), 3);
        } break;
        case ID_EDIT_IOU_THRES: {
            float v;
            if (TryParseFloat(GetDlgItem(hwnd, ID_EDIT_IOU_THRES), &v)) p.iou_thres.store(std::clamp(v, 0.0f, 1.0f));
            SetEditTextF(GetDlgItem(hwnd, ID_EDIT_IOU_THRES), p.iou_thres.load(), 3);
        } break;

        case ID_EDIT_MAX_MISS: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_MAX_MISS), &v)) p.max_miss.store(std::max(0, v));
            SetEditTextI(GetDlgItem(hwnd, ID_EDIT_MAX_MISS), p.max_miss.load());
        } break;
        case ID_EDIT_MAX_REASSOC_PX: {
            float v;
            if (TryParseFloat(GetDlgItem(hwnd, ID_EDIT_MAX_REASSOC_PX), &v)) p.max_reassoc_px.store(std::max(0.0f, v));
            SetEditTextF(GetDlgItem(hwnd, ID_EDIT_MAX_REASSOC_PX), p.max_reassoc_px.load(), 2);
        } break;
        case ID_EDIT_AIM_UP_RATIO: {
            float v;
            if (TryParseFloat(GetDlgItem(hwnd, ID_EDIT_AIM_UP_RATIO), &v)) p.aim_up_ratio.store(std::clamp(v, 0.0f, 1.0f));
            SetEditTextF(GetDlgItem(hwnd, ID_EDIT_AIM_UP_RATIO), p.aim_up_ratio.load(), 3);
        } break;
        case ID_EDIT_RESIDUAL_SIZE_RATIO: {
            float v;
            if (TryParseFloat(GetDlgItem(hwnd, ID_EDIT_RESIDUAL_SIZE_RATIO), &v)) p.residual_size_ratio.store(std::max(1.0f, v));
            SetEditTextF(GetDlgItem(hwnd, ID_EDIT_RESIDUAL_SIZE_RATIO), p.residual_size_ratio.load(), 2);
        } break;
        case ID_EDIT_STABLE_RATIO: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_STABLE_RATIO), &v)) {
                // clamp to [1, max_miss] to avoid useless values
                int mm = std::max(1, p.max_miss.load());
                p.stable_ratio.store(std::clamp(v, 1, mm));
            }
            SetEditTextI(GetDlgItem(hwnd, ID_EDIT_STABLE_RATIO), p.stable_ratio.load());
        } break;

        case ID_EDIT_ATTACH_DEADZONE: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_ATTACH_DEADZONE), &v)) p.attachDeadzonePx.store(std::max(0, v));
            SetEditTextI(GetDlgItem(hwnd, ID_EDIT_ATTACH_DEADZONE), p.attachDeadzonePx.load());
        } break;
        case ID_EDIT_ATTACH_GAIN: {
            float v;
            if (TryParseFloat(GetDlgItem(hwnd, ID_EDIT_ATTACH_GAIN), &v)) p.attachGain.store(std::max(0.0f, v));
            SetEditTextF(GetDlgItem(hwnd, ID_EDIT_ATTACH_GAIN), p.attachGain.load(), 3);
        } break;

        case ID_EDIT_ATTACH_MAXSTEP: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_ATTACH_MAXSTEP), &v)) p.attachMaxStepPx.store(std::clamp(v, 1, 127));
            SetEditTextI(GetDlgItem(hwnd, ID_EDIT_ATTACH_MAXSTEP), p.attachMaxStepPx.load());
        } break;

        case ID_EDIT_ATTACH_SMOOTHING: {
            float v;
            if (TryParseFloat(GetDlgItem(hwnd, ID_EDIT_ATTACH_SMOOTHING), &v)) p.attachSmoothing.store(std::max(0.0f, v));
            SetEditTextF(GetDlgItem(hwnd, ID_EDIT_ATTACH_SMOOTHING), p.attachSmoothing.load(), 3);
        } break;
        case ID_EDIT_FA_VID: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_FA_VID), &v)) p.mouse_fa_vid.store(v);
            SetEditTextIHex(GetDlgItem(hwnd, ID_EDIT_FA_VID), p.mouse_fa_vid.load());
        } break;
        case ID_EDIT_FA_PID: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_FA_PID), &v)) p.mouse_fa_pid.store(v);
            SetEditTextIHex(GetDlgItem(hwnd, ID_EDIT_FA_PID), p.mouse_fa_pid.load());
        } break;
        case ID_EDIT_FA_HEAD: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_FA_HEAD), &v)) p.mouse_fa_head.store(v);
            SetEditTextIHex(GetDlgItem(hwnd, ID_EDIT_FA_HEAD), p.mouse_fa_head.load());
        } break;
        case ID_EDIT_FA_TAIL: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_FA_TAIL), &v)) p.mouse_fa_tail.store(v);
            SetEditTextIHex(GetDlgItem(hwnd, ID_EDIT_FA_TAIL), p.mouse_fa_tail.load());
        } break;
        case ID_EDIT_FA_RID: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_FA_RID), &v)) p.mouse_fa_report_id.store(v);
            SetEditTextIHex(GetDlgItem(hwnd, ID_EDIT_FA_RID), p.mouse_fa_report_id.load());
        } break;
        case ID_EDIT_FA_RECONN: {
            int v;
            if (TryParseIntDecOrHex(GetDlgItem(hwnd, ID_EDIT_FA_RECONN), &v)) p.mouse_fa_reconnectMs.store(std::max(0, v));
            SetEditTextI(GetDlgItem(hwnd, ID_EDIT_FA_RECONN), p.mouse_fa_reconnectMs.load());
        } break;

        default: break;
        }
    }

    // Edit 子类：Enter 触发 Apply；Esc 回滚（回滚=重新从 params 填回）
    static LRESULT CALLBACK EditSubclassProc(HWND hEdit, UINT msg, WPARAM wParam, LPARAM lParam,
        UINT_PTR, DWORD_PTR) {
        if (msg == WM_KEYDOWN) {
            if (wParam == VK_RETURN) {
                HWND parent = GetParent(hEdit);
                int id = GetDlgCtrlID(hEdit);
                PostMessageW(parent, WM_APP_APPLY_EDIT, (WPARAM)id, 0);
                return 0;
            }
            if (wParam == VK_ESCAPE) {
                // 让父窗口按 params 回填
                HWND parent = GetParent(hEdit);
                int id = GetDlgCtrlID(hEdit);
                PostMessageW(parent, WM_APP_APPLY_EDIT, (WPARAM)id, 1 /*rollback*/);
                return 0;
            }
        }
        return DefSubclassProc(hEdit, msg, wParam, lParam);
    }

    static void SubclassEdit(HWND hEdit) {
        if (!hEdit) return;
        SetWindowSubclass(hEdit, EditSubclassProc, 1, 0);
    }

    static void RollbackEditById(HWND hwnd, int id) {
        UiState* st = GetState(hwnd);
        if (!st || !st->params) return;
        RuntimeParams& p = *st->params;

        switch (id) {
        case ID_EDIT_CONF_THRES:      SetEditTextF(GetDlgItem(hwnd, id), p.conf_thres.load(), 3); break;
        case ID_EDIT_IOU_THRES:       SetEditTextF(GetDlgItem(hwnd, id), p.iou_thres.load(), 3); break;

        case ID_EDIT_MAX_MISS:        SetEditTextI(GetDlgItem(hwnd, id), p.max_miss.load()); break;
        case ID_EDIT_MAX_REASSOC_PX:  SetEditTextF(GetDlgItem(hwnd, id), p.max_reassoc_px.load(), 2); break;
        case ID_EDIT_AIM_UP_RATIO:    SetEditTextF(GetDlgItem(hwnd, id), p.aim_up_ratio.load(), 3); break;
        case ID_EDIT_RESIDUAL_SIZE_RATIO: SetEditTextF(GetDlgItem(hwnd, id), p.residual_size_ratio.load(), 2); break;
        case ID_EDIT_STABLE_RATIO:   SetEditTextI(GetDlgItem(hwnd, id), p.stable_ratio.load()); break;

        case ID_EDIT_ATTACH_DEADZONE: SetEditTextI(GetDlgItem(hwnd, id), p.attachDeadzonePx.load()); break;
        case ID_EDIT_ATTACH_GAIN:     SetEditTextF(GetDlgItem(hwnd, id), p.attachGain.load(), 3); break;
        case ID_EDIT_ATTACH_MAXSTEP:  SetEditTextI(GetDlgItem(hwnd, id), p.attachMaxStepPx.load()); break;
        case ID_EDIT_ATTACH_SMOOTHING:SetEditTextF(GetDlgItem(hwnd, id), p.attachSmoothing.load(), 3); break;

        case ID_EDIT_FA_VID:          SetEditTextIHex(GetDlgItem(hwnd, id), p.mouse_fa_vid.load()); break;
        case ID_EDIT_FA_PID:          SetEditTextIHex(GetDlgItem(hwnd, id), p.mouse_fa_pid.load()); break;
        case ID_EDIT_FA_HEAD:         SetEditTextIHex(GetDlgItem(hwnd, id), p.mouse_fa_head.load()); break;
        case ID_EDIT_FA_TAIL:         SetEditTextIHex(GetDlgItem(hwnd, id), p.mouse_fa_tail.load()); break;
        case ID_EDIT_FA_RID:          SetEditTextIHex(GetDlgItem(hwnd, id), p.mouse_fa_report_id.load()); break;
        case ID_EDIT_FA_RECONN:       SetEditTextI(GetDlgItem(hwnd, id), p.mouse_fa_reconnectMs.load()); break;
        default: break;
        }
    }

    static void DrawPanel(HDC hdc, const RECT& rc, const wchar_t* title) {
        // 虚线圆角框
        HPEN pen = CreatePen(PS_DOT, 1, GetSysColor(COLOR_WINDOWTEXT));
        HGDIOBJ oldPen = SelectObject(hdc, pen);
        HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));

        RoundRect(hdc, rc.left, rc.top, rc.right, rc.bottom, 10, 10);

        SelectObject(hdc, oldBrush);
        SelectObject(hdc, oldPen);
        DeleteObject(pen);

        // 标题
        RECT rt = rc;
        rt.left += 10;
        rt.top += 4;
        SetBkMode(hdc, TRANSPARENT);
        DrawTextW(hdc, title, -1, &rt, DT_LEFT | DT_TOP | DT_SINGLELINE);
    }

    static void LayoutPanels(HWND hwnd, UiState* st) {
        RECT rc{};
        GetClientRect(hwnd, &rc);
        int W = rc.right - rc.left;
        int H = rc.bottom - rc.top;

        int margin = kMargin;
        int gap = kGap;

        // 顶部留一行给置顶
        int topBarH = 34;

        int usableW = W - margin * 2 - gap;
        int colW = std::max(200, usableW / 2);
        int leftX = margin;
        int rightX = margin + colW + gap;

        int y0 = margin + topBarH;
        int bottom = H - margin;

        // 左列：调试 / 状态 / Fa
        int leftH = bottom - y0;
        int dbgH = std::max(220, (int)(leftH * 0.42f));
        int statusH = 90;
        int faH = std::max(160, leftH - dbgH - statusH - gap * 2);

        st->rcDbg = { leftX, y0, leftX + colW, y0 + dbgH };
        st->rcStatus = { leftX, st->rcDbg.bottom + gap, leftX + colW, st->rcDbg.bottom + gap + statusH };
        st->rcFa = { leftX, st->rcStatus.bottom + gap, leftX + colW, bottom };

        // 右列：检测 / Tracking / 鼠标
        int rightH = bottom - y0;
        int detH = 120;
        int trackH = std::max(160, (int)(rightH * 0.40f));
        st->rcDetect = { rightX, y0, rightX + colW, y0 + detH };
        st->rcTrack = { rightX, st->rcDetect.bottom + gap, rightX + colW, st->rcDetect.bottom + gap + trackH };
        st->rcMouse = { rightX, st->rcTrack.bottom + gap, rightX + colW, bottom };
    }

    static void LayoutControls(HWND hwnd, UiState* st) {
        LayoutPanels(hwnd, st);

        // 顶部：置顶
        MoveWindow(st->chkTopmost, kMargin, kMargin, 120, 24, TRUE);

        // 调试区：两列 label + value
        int x = st->rcDbg.left + 12;
        int y = st->rcDbg.top + 30;
        int valueX = x + kLblW;

        struct Row { const wchar_t* name; int valId; };
        Row rows[] = {
            { L"\u6355\u83b7\u5e27\u7387", ID_LBL_CAP_FPS_VAL },
            { L"\u63a8\u7406\u5e27\u7387", ID_LBL_INF_FPS_VAL },
            { L"det \u6570\u91cf", ID_LBL_DET_CNT_VAL },
            { L"track \u72b6\u6001", ID_LBL_TRACK_VAL },
            { L"dx/dy \u8f93\u51fa", ID_LBL_DXDY_VAL },
            { L"FaHID \u5df2\u8fde\u63a5", ID_LBL_FAHID_CONN_VAL },
            { L"FaHID active", ID_LBL_FAHID_ACT_VAL },
            { L"\u6293\u5e27/\u9ed1\u5c4f\u6062\u590d", ID_LBL_CAPTURE_ERR_VAL },
        };

        for (int i = 0; i < (int)(sizeof(rows) / sizeof(rows[0])); ++i) {
            HWND hName = GetDlgItem(hwnd, 10000 + i); // 运行时创建的 label id（我们不需要处理消息）
            // 如果 label 不存在就跳过（安全）
            if (hName) MoveWindow(hName, x, y, kLblW, kRowH, TRUE);

            HWND hVal = GetDlgItem(hwnd, rows[i].valId);
            if (hVal) MoveWindow(hVal, valueX, y, (st->rcDbg.right - valueX - 12), kRowH, TRUE);

            y += kRowH + 4;
        }

        // 状态区：两行只读勾选
        int sx = st->rcStatus.left + 12;
        int sy = st->rcStatus.top + 30;
        MoveWindow(st->chkMaster, sx, sy, (st->rcStatus.right - sx - 12), 24, TRUE);
        MoveWindow(st->chkFaHid, sx, sy + 28, (st->rcStatus.right - sx - 12), 24, TRUE);

        // Fa(HID) 参数：3行 x 2列
        int fx = st->rcFa.left + 12;
        int fy = st->rcFa.top + 30;
        int colGap = 18;
        int col1x = fx;
        int col2x = fx + kLblW + kEditW + colGap;

        auto placePair = [&](int colX, int rowY, const wchar_t* label, HWND edit, int lblRuntimeId) {
            HWND hLbl = GetDlgItem(hwnd, lblRuntimeId);
            if (hLbl) {
                MoveWindow(hLbl, colX, rowY, kLblW, kRowH, TRUE);
                MoveWindow(edit, colX + kLblW, rowY, kEditW, kRowH, TRUE);
            }
            };

        placePair(col1x, fy, L"VID", st->eFaVid, 11000);
        placePair(col2x, fy, L"PID", st->eFaPid, 11001);
        placePair(col1x, fy + 30, L"Head", st->eFaHead, 11002);
        placePair(col2x, fy + 30, L"Tail", st->eFaTail, 11003);
        placePair(col1x, fy + 60, L"RID", st->eFaRid, 11004);
        placePair(col2x, fy + 60, L"\u91cd\u8fde(ms)", st->eFaReconn, 11005);

        // 检测参数
        int dx = st->rcDetect.left + 12;
        int dy = st->rcDetect.top + 30;
        HWND lblConf = GetDlgItem(hwnd, 12000);
        HWND lblIou = GetDlgItem(hwnd, 12001);
        if (lblConf) { MoveWindow(lblConf, dx, dy, kLblW, kRowH, TRUE); MoveWindow(st->eConf, dx + kLblW, dy, kEditW, kRowH, TRUE); }
        if (lblIou) { MoveWindow(lblIou, dx, dy + 30, kLblW, kRowH, TRUE); MoveWindow(st->eIou, dx + kLblW, dy + 30, kEditW, kRowH, TRUE); }

        // Tracking
        int tx = st->rcTrack.left + 12;
        int ty = st->rcTrack.top + 30;
        HWND lblMM = GetDlgItem(hwnd, 13000);
        HWND lblMR = GetDlgItem(hwnd, 13001);
        HWND lblAU = GetDlgItem(hwnd, 13002);
        HWND lblSR = GetDlgItem(hwnd, 13003);
        HWND lblST = GetDlgItem(hwnd, 13004);
        if (lblMM) { MoveWindow(lblMM, tx, ty, kLblW, kRowH, TRUE); MoveWindow(st->eMaxMiss, tx + kLblW, ty, kEditW, kRowH, TRUE); }
        if (lblMR) { MoveWindow(lblMR, tx, ty + 30, kLblW, kRowH, TRUE); MoveWindow(st->eMaxReassoc, tx + kLblW, ty + 30, kEditW, kRowH, TRUE); }
        if (lblAU) { MoveWindow(lblAU, tx, ty + 60, kLblW, kRowH, TRUE); MoveWindow(st->eAimUp, tx + kLblW, ty + 60, kEditW, kRowH, TRUE); }
        if (lblSR) { MoveWindow(lblSR, tx, ty + 90, kLblW, kRowH, TRUE); MoveWindow(st->eSizeRatio, tx + kLblW, ty + 90, kEditW, kRowH, TRUE); }
        if (lblST) { MoveWindow(lblST, tx, ty + 120, kLblW, kRowH, TRUE); MoveWindow(st->eStableNeed, tx + kLblW, ty + 120, kEditW, kRowH, TRUE); }

        // 鼠标控制
        int mx = st->rcMouse.left + 12;
        int my = st->rcMouse.top + 30;
        HWND lblAD = GetDlgItem(hwnd, 14000);
        HWND lblAG = GetDlgItem(hwnd, 14001);
        HWND lblAM = GetDlgItem(hwnd, 14006); // 新增：最大步长
        HWND lblAS = GetDlgItem(hwnd, 14002);

        if (lblAD) { MoveWindow(lblAD, mx, my, kLblW, kRowH, TRUE); MoveWindow(st->eAttachDeadzone, mx + kLblW, my, kEditW, kRowH, TRUE); }
        if (lblAG) { MoveWindow(lblAG, mx, my + 30, kLblW, kRowH, TRUE); MoveWindow(st->eAttachGain, mx + kLblW, my + 30, kEditW, kRowH, TRUE); }
        if (lblAM) { MoveWindow(lblAM, mx, my + 60, kLblW, kRowH, TRUE); MoveWindow(st->eAttachMaxStep, mx + kLblW, my + 60, kEditW, kRowH, TRUE); }
        if (lblAS) { MoveWindow(lblAS, mx, my + 90, kLblW, kRowH, TRUE); MoveWindow(st->eAttachSmooth, mx + kLblW, my + 90, kEditW, kRowH, TRUE); }

        // 下面三项是旧的动态策略 UI（你后续要删的话，这里也一起删）
    }

    static void RefreshUi(HWND hwnd, UiState* st) {
        if (!st || !st->params) return;
        RuntimeParams& p = *st->params;

        // 只读状态展示
        SendMessageW(st->chkMaster, BM_SETCHECK, p.enabled.load() ? BST_CHECKED : BST_UNCHECKED, 0);
        SendMessageW(st->chkFaHid, BM_SETCHECK, p.mouse_useFaHid.load() ? BST_CHECKED : BST_UNCHECKED, 0);

        // 调试区：从 BackendWorker 拉取快照（UI 与后端解耦）
        if (st->worker) {
            const auto s = st->worker->getUiStats();

            wchar_t buf[256]{};

            // cap / infer fps
            std::swprintf(buf, 256, L"%.1f", s.cap_fps);
            SetWindowTextW(st->vCapFps, buf);

            std::swprintf(buf, 256, L"%.1f", s.infer_fps);
            SetWindowTextW(st->vInfFps, buf);

            // det
            std::swprintf(buf, 256, L"%d", s.det_count);
            SetWindowTextW(st->vDetCnt, buf);

            // track 状态（中文）
            if (!s.lock_has) {
                std::swprintf(buf, 256, L"未锁定 | 轨迹:%d | lockedUid:%d", s.track_count, s.locked_uid);
            }
            else if (s.lock_residual) {
                std::swprintf(buf, 256, L"残影期(不移动) | uid:%d miss:%d | 轨迹:%d", s.lock_uid, s.lock_miss, s.track_count);
            }
            else {
                std::swprintf(buf, 256, L"锁定 | uid:%d miss:%d rank:%d | 轨迹:%d", s.lock_uid, s.lock_miss, s.lock_srank, s.track_count);
            }
            SetWindowTextW(st->vTrack, buf);

            // dx/dy
            std::swprintf(buf, 256, L"%d / %d", s.dx, s.dy);
            SetWindowTextW(st->vDxDy, buf);

            // FaHID
            SetWindowTextW(st->vFaConn, s.fa_connected ? L"已连接" : L"未连接");
            SetWindowTextW(st->vFaAct, s.fa_active ? L"Active" : L"未激活");

            // capture 错误/恢复/权限提示
            if (s.cap_last_hr == 0 && s.cap_fail_streak == 0) {
                std::swprintf(buf, 256, L"正常");
            }
            else {
                std::swprintf(buf, 256, L"hr=0x%08X | 连续失败:%d | 黑屏:%d | 自动恢复:%d%s",
                    (unsigned)s.cap_last_hr,
                    s.cap_fail_streak,
                    s.cap_black_streak,
                    s.cap_recover_count,
                    s.cap_permission_hint ? L" | 可能缺少权限/被保护窗口" : L"");
            }
            SetWindowTextW(st->vCapErr, buf);
        }
        else {
            SetWindowTextW(st->vCapFps, L"—");
            SetWindowTextW(st->vInfFps, L"—");
            SetWindowTextW(st->vDetCnt, L"—");
            SetWindowTextW(st->vTrack, L"—");
            SetWindowTextW(st->vDxDy, L"—");
            SetWindowTextW(st->vFaConn, L"—");
            SetWindowTextW(st->vFaAct, L"—");
            SetWindowTextW(st->vCapErr, L"—");
        }
    }

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        UiState* st = GetState(hwnd);

        switch (msg) {
        case WM_CREATE: {
            auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
            UiState* s = new UiState();
            UiCreateArgs* args = reinterpret_cast<UiCreateArgs*>(cs->lpCreateParams);
            s->params = args ? args->params : nullptr;
            s->worker = args ? args->worker : nullptr;
            // lpCreateParams 是 CreateMainWindow new 出来的，WM_CREATE 里读完就释放
            if (args) delete args;

            SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)s);
            st = s;

            // common controls
            INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_STANDARD_CLASSES };
            InitCommonControlsEx(&icc);

            st->font = CreateUiFont(hwnd);

            st->chkTopmost = MakeCheck(hwnd, ID_CHK_TOPMOST, L"\u7a97\u53e3\u7f6e\u9876");
            SetCtlFont(st->chkTopmost, st->font);

            // 调试区：创建 name labels（运行时 id 10000+i）+ value labels（固定 id）
            const wchar_t* dbgNames[] = {
                L"\u6355\u83b7\u5e27\u7387", L"\u63a8\u7406\u5e27\u7387", L"det \u6570\u91cf", L"track \u72b6\u6001",
                L"dx/dy \u8f93\u51fa", L"FaHID \u5df2\u8fde\u63a5", L"FaHID active", L"\u6293\u5e27/\u9ed1\u5c4f\u6062\u590d"
            };
            for (int i = 0; i < 8; ++i) {
                HWND h = CreateWindowExW(0, L"STATIC", dbgNames[i], WS_CHILD | WS_VISIBLE,
                    0, 0, 10, 10, hwnd, (HMENU)(INT_PTR)(10000 + i), nullptr, nullptr);
                SetCtlFont(h, st->font);
            }
            st->vCapFps = MakeValue(hwnd, ID_LBL_CAP_FPS_VAL); SetCtlFont(st->vCapFps, st->font);
            st->vInfFps = MakeValue(hwnd, ID_LBL_INF_FPS_VAL); SetCtlFont(st->vInfFps, st->font);
            st->vDetCnt = MakeValue(hwnd, ID_LBL_DET_CNT_VAL); SetCtlFont(st->vDetCnt, st->font);
            st->vTrack = MakeValue(hwnd, ID_LBL_TRACK_VAL); SetCtlFont(st->vTrack, st->font);
            st->vDxDy = MakeValue(hwnd, ID_LBL_DXDY_VAL); SetCtlFont(st->vDxDy, st->font);
            st->vFaConn = MakeValue(hwnd, ID_LBL_FAHID_CONN_VAL); SetCtlFont(st->vFaConn, st->font);
            st->vFaAct = MakeValue(hwnd, ID_LBL_FAHID_ACT_VAL); SetCtlFont(st->vFaAct, st->font);
            st->vCapErr = MakeValue(hwnd, ID_LBL_CAPTURE_ERR_VAL); SetCtlFont(st->vCapErr, st->font);

            // 状态区
            st->chkMaster = MakeCheck(hwnd, ID_CHK_MASTER_STATUS, L"\u9879\u76ee\u603b\u5f00\u5173(PageUp)\uff1a\u5df2\u542f\u52a8");
            st->chkFaHid = MakeCheck(hwnd, ID_CHK_FAHID_STATUS, L"\u786c\u4ef6HID(FaHID)\uff1a\u5df2\u542f\u7528");
            SetCtlFont(st->chkMaster, st->font);
            SetCtlFont(st->chkFaHid, st->font);
            SetReadOnlyCheck(st->chkMaster);
            SetReadOnlyCheck(st->chkFaHid);

            // Fa(HID) labels
            MakeLabel(hwnd, 11000, L"VID"); MakeLabel(hwnd, 11001, L"PID");
            MakeLabel(hwnd, 11002, L"Head"); MakeLabel(hwnd, 11003, L"Tail");
            MakeLabel(hwnd, 11004, L"RID"); MakeLabel(hwnd, 11005, L"\u91cd\u8fde(ms)");
            for (int id = 11000; id <= 11005; ++id) SetCtlFont(GetDlgItem(hwnd, id), st->font);

            st->eFaVid = MakeEdit(hwnd, ID_EDIT_FA_VID); SetCtlFont(st->eFaVid, st->font); SubclassEdit(st->eFaVid);
            st->eFaPid = MakeEdit(hwnd, ID_EDIT_FA_PID); SetCtlFont(st->eFaPid, st->font); SubclassEdit(st->eFaPid);
            st->eFaHead = MakeEdit(hwnd, ID_EDIT_FA_HEAD); SetCtlFont(st->eFaHead, st->font); SubclassEdit(st->eFaHead);
            st->eFaTail = MakeEdit(hwnd, ID_EDIT_FA_TAIL); SetCtlFont(st->eFaTail, st->font); SubclassEdit(st->eFaTail);
            st->eFaRid = MakeEdit(hwnd, ID_EDIT_FA_RID); SetCtlFont(st->eFaRid, st->font); SubclassEdit(st->eFaRid);
            st->eFaReconn = MakeEdit(hwnd, ID_EDIT_FA_RECONN); SetCtlFont(st->eFaReconn, st->font); SubclassEdit(st->eFaReconn);

            // 检测 labels + edits
            MakeLabel(hwnd, 12000, L"\u68c0\u6d4b\u9608\u503c(conf)");
            MakeLabel(hwnd, 12001, L"IOU \u9608\u503c(iou)");
            SetCtlFont(GetDlgItem(hwnd, 12000), st->font);
            SetCtlFont(GetDlgItem(hwnd, 12001), st->font);
            st->eConf = MakeEdit(hwnd, ID_EDIT_CONF_THRES); SetCtlFont(st->eConf, st->font); SubclassEdit(st->eConf);
            st->eIou = MakeEdit(hwnd, ID_EDIT_IOU_THRES);  SetCtlFont(st->eIou, st->font); SubclassEdit(st->eIou);

            // Tracking labels + edits
            MakeLabel(hwnd, 13000, L"\u6700\u5927\u4e22\u5931\u5e27(max_miss)");
            MakeLabel(hwnd, 13001, L"\u91cd\u5173\u8054\u8ddd\u79bb(px)");
            MakeLabel(hwnd, 13002, L"\u5411\u4e0a\u7784\u51c6\u6bd4\u4f8b");
            MakeLabel(hwnd, 13003, L"\u5c3a\u5bf8\u76f8\u4f3c\u6bd4\u4f8b");
            MakeLabel(hwnd, 13004, L"\u7a33\u5b9a\u5339\u914d\u5e27\u6570");
            SetCtlFont(GetDlgItem(hwnd, 13000), st->font);
            SetCtlFont(GetDlgItem(hwnd, 13001), st->font);
            SetCtlFont(GetDlgItem(hwnd, 13002), st->font);
            SetCtlFont(GetDlgItem(hwnd, 13003), st->font);
            SetCtlFont(GetDlgItem(hwnd, 13004), st->font);
            st->eMaxMiss = MakeEdit(hwnd, ID_EDIT_MAX_MISS); SetCtlFont(st->eMaxMiss, st->font); SubclassEdit(st->eMaxMiss);
            st->eMaxReassoc = MakeEdit(hwnd, ID_EDIT_MAX_REASSOC_PX); SetCtlFont(st->eMaxReassoc, st->font); SubclassEdit(st->eMaxReassoc);
            st->eAimUp = MakeEdit(hwnd, ID_EDIT_AIM_UP_RATIO); SetCtlFont(st->eAimUp, st->font); SubclassEdit(st->eAimUp);
            st->eSizeRatio = MakeEdit(hwnd, ID_EDIT_RESIDUAL_SIZE_RATIO); SetCtlFont(st->eSizeRatio, st->font); SubclassEdit(st->eSizeRatio);
            st->eStableNeed = MakeEdit(hwnd, ID_EDIT_STABLE_RATIO); SetCtlFont(st->eStableNeed, st->font); SubclassEdit(st->eStableNeed);

            // 鼠标 labels + edits
            MakeLabel(hwnd, 14000, L"吸附死区(px)");
            MakeLabel(hwnd, 14001, L"吸附增益");
            MakeLabel(hwnd, 14006, L"吸附最大步长(px)");
            MakeLabel(hwnd, 14002, L"吸附平滑(0~1)");
            for (int id : { 14000, 14001, 14002, 14006 }) {
                SetCtlFont(GetDlgItem(hwnd, id), st->font);
            }

            st->eAttachDeadzone = MakeEdit(hwnd, ID_EDIT_ATTACH_DEADZONE); SetCtlFont(st->eAttachDeadzone, st->font); SubclassEdit(st->eAttachDeadzone);
            st->eAttachGain = MakeEdit(hwnd, ID_EDIT_ATTACH_GAIN); SetCtlFont(st->eAttachGain, st->font); SubclassEdit(st->eAttachGain);
            st->eAttachMaxStep = MakeEdit(hwnd, ID_EDIT_ATTACH_MAXSTEP); SetCtlFont(st->eAttachMaxStep, st->font); SubclassEdit(st->eAttachMaxStep);
            st->eAttachSmooth = MakeEdit(hwnd, ID_EDIT_ATTACH_SMOOTHING); SetCtlFont(st->eAttachSmooth, st->font); SubclassEdit(st->eAttachSmooth);

            // 初始回填
            RollbackEditById(hwnd, ID_EDIT_CONF_THRES);
            RollbackEditById(hwnd, ID_EDIT_IOU_THRES);
            RollbackEditById(hwnd, ID_EDIT_MAX_MISS);
            RollbackEditById(hwnd, ID_EDIT_MAX_REASSOC_PX);
            RollbackEditById(hwnd, ID_EDIT_AIM_UP_RATIO);
            RollbackEditById(hwnd, ID_EDIT_RESIDUAL_SIZE_RATIO);
            RollbackEditById(hwnd, ID_EDIT_STABLE_RATIO);
            RollbackEditById(hwnd, ID_EDIT_ATTACH_DEADZONE);
            RollbackEditById(hwnd, ID_EDIT_ATTACH_GAIN);
            RollbackEditById(hwnd, ID_EDIT_ATTACH_MAXSTEP);
            RollbackEditById(hwnd, ID_EDIT_ATTACH_SMOOTHING);
            RollbackEditById(hwnd, ID_EDIT_FA_VID);
            RollbackEditById(hwnd, ID_EDIT_FA_PID);
            RollbackEditById(hwnd, ID_EDIT_FA_HEAD);
            RollbackEditById(hwnd, ID_EDIT_FA_TAIL);
            RollbackEditById(hwnd, ID_EDIT_FA_RID);
            RollbackEditById(hwnd, ID_EDIT_FA_RECONN);

            SetTimer(hwnd, kUiTimerId, kUiTimerMs, nullptr);
            return 0;
        }

        case WM_SIZE: {
            if (st) LayoutControls(hwnd, st);
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;
        }

        case WM_COMMAND: {
            int id = LOWORD(wParam);
            int code = HIWORD(wParam);

            if (id == ID_CHK_TOPMOST && code == BN_CLICKED && st) {
                st->topmost = (SendMessageW(st->chkTopmost, BM_GETCHECK, 0, 0) == BST_CHECKED);
                SetWindowPos(hwnd, st->topmost ? HWND_TOPMOST : HWND_NOTOPMOST,
                    0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
                return 0;
            }

            // 失焦自动应用
            if (code == EN_KILLFOCUS) {
                ApplyEditById(hwnd, id);
                return 0;
            }
            return 0;
        }

        case WM_APP_APPLY_EDIT: {
            int id = (int)wParam;
            bool rollback = (lParam != 0);
            if (rollback) RollbackEditById(hwnd, id);
            else ApplyEditById(hwnd, id);
            return 0;
        }

        case WM_TIMER: {
            if (wParam == kUiTimerId && st) RefreshUi(hwnd, st);
            return 0;
        }

        case WM_PAINT: {
            PAINTSTRUCT ps{};
            HDC hdc = BeginPaint(hwnd, &ps);
            if (st) {
                DrawPanel(hdc, st->rcDbg, L"\u8c03\u8bd5\u4fe1\u606f");
                DrawPanel(hdc, st->rcStatus, L"\u6309\u94ae\u4e0e\u72b6\u6001");
                DrawPanel(hdc, st->rcFa, L"Fa(HID) \u53c2\u6570");
                DrawPanel(hdc, st->rcDetect, L"\u68c0\u6d4b\u53c2\u6570");
                DrawPanel(hdc, st->rcTrack, L"Tracking \u53c2\u6570");
                DrawPanel(hdc, st->rcMouse, L"\u9f20\u6807\u63a7\u5236\u53c2\u6570");
            }
            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_GETMINMAXINFO: {
            // 限定最小大小；最大不超过工作区
            MINMAXINFO* mmi = reinterpret_cast<MINMAXINFO*>(lParam);
            mmi->ptMinTrackSize.x = 900;
            mmi->ptMinTrackSize.y = 650;

            RECT work{};
            SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
            mmi->ptMaxTrackSize.x = work.right - work.left;
            mmi->ptMaxTrackSize.y = work.bottom - work.top;
            return 0;
        }

        case WM_DESTROY: {
            if (st) {
                KillTimer(hwnd, kUiTimerId);
                if (st->font) DeleteObject(st->font);
                delete st;
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            }
            PostQuitMessage(0);
            return 0;
        }

        default:
            return DefWindowProcW(hwnd, msg, wParam, lParam);
        }
    }

} // namespace

HWND CreateMainWindow(HINSTANCE hInst, RuntimeParams* params, BackendWorker* worker) {
    const wchar_t* kCls = L"DD_TRY_YOLO_UI";

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.hInstance = hInst;
    wc.lpszClassName = kCls;
    wc.lpfnWndProc = WndProc;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClassExW(&wc);

    DWORD style = WS_OVERLAPPEDWINDOW;
    UiCreateArgs* args = new UiCreateArgs();
    args->params = params;
    args->worker = worker;

    HWND hwnd = CreateWindowExW(0, kCls, L"\u4eba\u773c\u63a7\u9f20\u6807 - \u53c2\u6570\u9762\u677f",
        style,
        CW_USEDEFAULT, CW_USEDEFAULT, 1100, 760,
        nullptr, nullptr, hInst, args);

    if (!hwnd) {
        delete args;
        return nullptr;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    return hwnd;
}

