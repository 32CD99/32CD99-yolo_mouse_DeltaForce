#pragma once
#include <cuda_runtime_api.h>

bool CheckCuda(cudaError_t e, const char* what);
