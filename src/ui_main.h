#pragma once
#include <windows.h>

struct RuntimeParams;
class BackendWorker;

// 创建主窗口并绑定 params/worker（UI 只直接读写 RuntimeParams；调试信息后续通过 worker 导出 stats 再接）
HWND CreateMainWindow(HINSTANCE hInst, RuntimeParams* params, BackendWorker* worker);
