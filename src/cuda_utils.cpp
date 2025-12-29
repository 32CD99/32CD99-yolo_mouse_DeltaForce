#include "cuda_utils.h"
#include <windows.h>

bool CheckCuda(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    // 你想静默就不输出；想输出就 OutputDebugStringA
    OutputDebugStringA("[CUDA] ");
    OutputDebugStringA(what);
    OutputDebugStringA("\n");
    return false;
}
