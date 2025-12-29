#define WIN32_LEAN_AND_MEAN

#include "runtime_params.h"
#include "backend_worker.h"
#include "ui_main.h"
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")


struct TimerResolutionGuard {
    TimerResolutionGuard() { timeBeginPeriod(1); }
    ~TimerResolutionGuard() { timeEndPeriod(1); }
};


int WINAPI wWinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ PWSTR pCmdLine,
    _In_ int nCmdShow
) {
    (void)hPrevInstance; (void)pCmdLine; (void)nCmdShow;
    TimerResolutionGuard _timerRes;

    RuntimeParams params;
    BackendWorker worker(&params);

    HWND hWnd = CreateMainWindow(hInstance, &params, &worker);
    if (!hWnd) return 1;

    if (!worker.start()) {
        MessageBoxW(nullptr, L"BackendWorker::start() failed.\n请查看 dd_trt_yolo.log 或按我加的弹窗提示定位失败步骤。",
            L"dd_trt_yolo", MB_ICONERROR | MB_OK);
        return 2;
    }

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    worker.stop();
    return 0;
}
