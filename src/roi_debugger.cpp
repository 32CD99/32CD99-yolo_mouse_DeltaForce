// roi_debugger.cpp
#include "roi_debugger.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace {

struct State {
    bool inited = false;
    bool visible = true;

    HWND hwnd = nullptr;
    HINSTANCE hinst = nullptr;

    int imgW = 0;
    int imgH = 0;

    int ox = 0;
    int oy = 0;

    std::vector<uint8_t> bgra;     // imgW*imgH*4
    BITMAPINFO bmi{};

    std::vector<RoiDebugBox> boxes;

    ComPtr<ID3D11Texture2D> staging;
};

State g;

static COLORREF colorForCls(int cls) {
    // A few distinct colors (BGR for GDI pens). cls usually 0 in your case.
    static const COLORREF palette[] = {
        RGB(0, 255, 0),     // green
        RGB(255, 0, 0),     // red
        RGB(0, 128, 255),   // orange-ish
        RGB(255, 0, 255),   // magenta
        RGB(0, 255, 255),   // cyan
        RGB(255, 255, 0),   // yellow
    };
    const int n = (int)(sizeof(palette) / sizeof(palette[0]));
    if (cls < 0) cls = 0;
    return palette[cls % n];
}

static void ensureBmi(int w, int h) {
    g.imgW = w;
    g.imgH = h;
    g.bgra.resize((size_t)w * (size_t)h * 4);

    ZeroMemory(&g.bmi, sizeof(g.bmi));
    g.bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    g.bmi.bmiHeader.biWidth = w;
    // Negative height => top-down bitmap (so we don't need to flip)
    g.bmi.bmiHeader.biHeight = -h;
    g.bmi.bmiHeader.biPlanes = 1;
    g.bmi.bmiHeader.biBitCount = 32;
    g.bmi.bmiHeader.biCompression = BI_RGB;
}

static bool ensureStaging(ID3D11Device* dev, ID3D11Texture2D* src) {
    if (!dev || !src) return false;

    D3D11_TEXTURE2D_DESC desc{};
    src->GetDesc(&desc);

    // Make a CPU-readable staging texture of the same size/format.
    if (g.staging) {
        D3D11_TEXTURE2D_DESC sdesc{};
        g.staging->GetDesc(&sdesc);
        if (sdesc.Width == desc.Width && sdesc.Height == desc.Height && sdesc.Format == desc.Format)
            return true;
        g.staging.Reset();
    }

    D3D11_TEXTURE2D_DESC s{};
    s.Width = desc.Width;
    s.Height = desc.Height;
    s.MipLevels = 1;
    s.ArraySize = 1;
    s.Format = desc.Format;
    s.SampleDesc.Count = 1;
    s.SampleDesc.Quality = 0;
    s.Usage = D3D11_USAGE_STAGING;
    s.BindFlags = 0;
    s.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    s.MiscFlags = 0;

    HRESULT hr = dev->CreateTexture2D(&s, nullptr, g.staging.GetAddressOf());
    return SUCCEEDED(hr) && g.staging;
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CLOSE:
        ShowWindow(hwnd, SW_HIDE);
        g.visible = false;
        return 0;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            ShowWindow(hwnd, SW_HIDE);
            g.visible = false;
            return 0;
        }
        break;
    case WM_PAINT: {
        PAINTSTRUCT ps{};
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT rc{};
        GetClientRect(hwnd, &rc);
        // RECT fields are LONG; std::max requires identical types.
        const LONG cwL = (rc.right > rc.left) ? (rc.right - rc.left) : 1;
        const LONG chL = (rc.bottom > rc.top) ? (rc.bottom - rc.top) : 1;
        const int cw = (int)cwL;
        const int ch = (int)chL;

        // Draw image (ROI)
        if (!g.bgra.empty() && g.imgW > 0 && g.imgH > 0) {
            StretchDIBits(
                hdc,
                0, 0, cw, ch,
                0, 0, g.imgW, g.imgH,
                g.bgra.data(),
                &g.bmi,
                DIB_RGB_COLORS,
                SRCCOPY
            );

            // Overlay boxes (convert SCREEN->ROI coords by subtracting ox/oy; then scale to window)
            const float sx = (float)cw / (float)g.imgW;
            const float sy = (float)ch / (float)g.imgH;

            // Use transparent brush for rectangles
            HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(HOLLOW_BRUSH));

            for (const auto& b : g.boxes) {
                float rx1 = (b.x1 - (float)g.ox);
                float ry1 = (b.y1 - (float)g.oy);
                float rx2 = (b.x2 - (float)g.ox);
                float ry2 = (b.y2 - (float)g.oy);

                // clip to ROI bounds
                rx1 = std::max(0.0f, std::min(rx1, (float)g.imgW));
                rx2 = std::max(0.0f, std::min(rx2, (float)g.imgW));
                ry1 = std::max(0.0f, std::min(ry1, (float)g.imgH));
                ry2 = std::max(0.0f, std::min(ry2, (float)g.imgH));

                int x1 = (int)std::lround(rx1 * sx);
                int y1 = (int)std::lround(ry1 * sy);
                int x2 = (int)std::lround(rx2 * sx);
                int y2 = (int)std::lround(ry2 * sy);

                HPEN pen = CreatePen(PS_SOLID, 2, colorForCls(b.cls));
                HGDIOBJ oldPen = SelectObject(hdc, pen);

                Rectangle(hdc, x1, y1, x2, y2);

                // Draw a small center crosshair for this box
                int cx = (x1 + x2) / 2;
                int cy = (y1 + y2) / 2;
                MoveToEx(hdc, cx - 6, cy, nullptr);
                LineTo(hdc, cx + 6, cy);
                MoveToEx(hdc, cx, cy - 6, nullptr);
                LineTo(hdc, cx, cy + 6);

                // Optional label
                char buf[64];
                int n = snprintf(buf, sizeof(buf), "c%d %.2f", b.cls, b.conf);
                if (n > 0) {
                    SetBkMode(hdc, TRANSPARENT);
                    SetTextColor(hdc, colorForCls(b.cls));
                    TextOutA(hdc, x1 + 3, y1 + 3, buf, n);
                }

                SelectObject(hdc, oldPen);
                DeleteObject(pen);
            }

            SelectObject(hdc, oldBrush);
        }

        EndPaint(hwnd, &ps);
        return 0;
    }
    default:
        break;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

static bool createWindow(const wchar_t* title) {
    if (g.hwnd) return true;

    g.hinst = GetModuleHandleW(nullptr);

    const wchar_t* kClass = L"ROI_DEBUG_WINDOW_CLASS";
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = g.hinst;
    wc.lpszClassName = kClass;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClassExW(&wc);

    std::wstring wtitle = title ? title : L"ROI Debug";
    g.hwnd = CreateWindowExW(
        0, kClass, wtitle.c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        540, 540,
        nullptr, nullptr, g.hinst, nullptr);

    if (!g.hwnd) return false;

    ShowWindow(g.hwnd, SW_SHOW);
    UpdateWindow(g.hwnd);
    g.visible = true;
    return true;
}

} // namespace

extern "C" {

bool RoiDebug_Init(const wchar_t* title) {
    if (g.inited) return true;
    g.inited = true;
    return createWindow(title);
}

void RoiDebug_Show(bool show) {
    if (!g.inited) RoiDebug_Init(L"ROI Debug");
    g.visible = show;
    if (g.hwnd) ShowWindow(g.hwnd, show ? SW_SHOW : SW_HIDE);
}

void RoiDebug_Toggle(void) {
    RoiDebug_Show(!g.visible);
}

bool RoiDebug_IsVisible(void) {
    return g.visible && g.hwnd && IsWindowVisible(g.hwnd);
}

bool RoiDebug_Update(ID3D11Device* dev,
                     ID3D11DeviceContext* ctx,
                     ID3D11Texture2D* roiTex,
                     int roiW, int roiH,
                     int ox, int oy,
                     const RoiDebugBox* boxes,
                     int nBoxes) {
    if (!g.inited) {
        if (!RoiDebug_Init(L"ROI Debug")) return false;
    }
    if (!g.visible) return true; // ignore updates when hidden

    if (!dev || !ctx || !roiTex || roiW <= 0 || roiH <= 0) return false;

    if (g.imgW != roiW || g.imgH != roiH || g.bgra.empty()) {
        ensureBmi(roiW, roiH);
    }

    g.ox = ox;
    g.oy = oy;

    g.boxes.clear();
    if (boxes && nBoxes > 0) {
        g.boxes.assign(boxes, boxes + nBoxes);
    }

    if (!ensureStaging(dev, roiTex)) return false;

    // Copy GPU texture -> staging
    ctx->CopyResource(g.staging.Get(), roiTex);

    // Map staging and copy into g.bgra (row-by-row)
    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = ctx->Map(g.staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr) || !mapped.pData) return false;

    const uint8_t* src = (const uint8_t*)mapped.pData;
    const size_t dstStride = (size_t)g.imgW * 4;
    const size_t srcStride = (size_t)mapped.RowPitch;

    uint8_t* dst = g.bgra.data();
    for (int y = 0; y < g.imgH; ++y) {
        memcpy(dst + (size_t)y * dstStride, src + (size_t)y * srcStride, dstStride);
    }

    ctx->Unmap(g.staging.Get(), 0);

    // Trigger repaint
    if (g.hwnd) InvalidateRect(g.hwnd, nullptr, FALSE);
    return true;
}

void RoiDebug_Poll(void) {
    if (!g.hwnd) return;

    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

void RoiDebug_Shutdown(void) {
    g.staging.Reset();
    g.bgra.clear();
    g.boxes.clear();

    if (g.hwnd) {
        DestroyWindow(g.hwnd);
        g.hwnd = nullptr;
    }
    g.inited = false;
    g.visible = false;
}

} // extern "C"
