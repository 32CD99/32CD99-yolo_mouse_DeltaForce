#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Convert BGRA8 cudaArray (from D3D11 texture) to NCHW float32 in [0,1]
// dst: float* with size 1*3*roi*roi
void preprocess_bgra8_to_f32_nchw(
  cudaArray_t srcBGRA,
  int roi,
  float* dstF32,
  cudaStream_t stream
);
