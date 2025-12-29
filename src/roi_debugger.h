// roi_debugger.h
// Lightweight ROI visualizer (Win32/GDI) for MSVC C++17 projects.
// - Displays the captured ROI (BGRA8) in a small window
// - Draws detection boxes (given in SCREEN pixel coordinates) on top
// - No external deps (no OpenCV). Uses D3D11 staging copy + GDI StretchDIBits.
//
// Usage (typical):
//   RoiDebug_Init(L"ROI Debug");
//   // per-frame:
//   RoiDebug_Update(dev, ctx, roiTex, roiW, roiH, info.ox, info.oy, boxes, nBoxes);
//   RoiDebug_Poll(); // keep window responsive
//   // on exit:
//   RoiDebug_Shutdown();

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RoiDebugBox {
    float x1, y1, x2, y2;   // SCREEN coordinates (pixels)
    float conf;
    int   cls;
} RoiDebugBox;

// Create the debug window. Safe to call multiple times (re-init is ignored).
// title: window title (UTF-16). Pass nullptr to use default.
bool RoiDebug_Init(const wchar_t* title);

// Show/hide the debug window.
void RoiDebug_Show(bool show);

// Toggle show/hide.
void RoiDebug_Toggle(void);

// Returns whether window is currently visible.
bool RoiDebug_IsVisible(void);

// Update the displayed frame.
// roiTex: D3D11 texture containing the ROI in BGRA8 (DXGI_FORMAT_B8G8R8A8_UNORM or similar).
// roiW/roiH: ROI size in pixels (e.g., 416x416).
// ox/oy: ROI origin in SCREEN coordinates (pixels). Used to convert boxes from SCREEN->ROI coords.
// boxes: array of detection boxes in SCREEN coords. May be nullptr if nBoxes==0.
// NOTE: Call RoiDebug_Poll() regularly to process window messages.
bool RoiDebug_Update(ID3D11Device* dev,
                     ID3D11DeviceContext* ctx,
                     ID3D11Texture2D* roiTex,
                     int roiW, int roiH,
                     int ox, int oy,
                     const RoiDebugBox* boxes,
                     int nBoxes);

// Pump window messages without blocking. Call once per main-loop iteration.
void RoiDebug_Poll(void);

// Destroy debug window and release resources.
void RoiDebug_Shutdown(void);

#ifdef __cplusplus
} // extern "C"
#endif
