#include "preprocess.h"
#include <cuda_runtime.h>

__global__ void k_bgra_to_f32(cudaTextureObject_t tex, int roi, float* out)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= roi || y >= roi) return;

  uchar4 bgra = tex2D<uchar4>(tex, (float)x + 0.5f, (float)y + 0.5f);

  float b = (float)bgra.x * (1.0f/255.0f);
  float g = (float)bgra.y * (1.0f/255.0f);
  float r = (float)bgra.z * (1.0f/255.0f);

  int hw = roi * roi;
  int idx = y * roi + x;

  out[0*hw + idx] = r;
  out[1*hw + idx] = g;
  out[2*hw + idx] = b;
}

void preprocess_bgra8_to_f32_nchw(
  cudaArray_t srcBGRA,
  int roi,
  float* dstF32,
  cudaStream_t stream)
{
  cudaResourceDesc res{};
  res.resType = cudaResourceTypeArray;
  res.res.array.array = srcBGRA;

  cudaTextureDesc td{};
  td.addressMode[0] = cudaAddressModeClamp;
  td.addressMode[1] = cudaAddressModeClamp;
  td.filterMode = cudaFilterModePoint;
  td.readMode = cudaReadModeElementType;
  td.normalizedCoords = 0;

  cudaTextureObject_t tex = 0;
  cudaCreateTextureObject(&tex, &res, &td, nullptr);

  dim3 block(16,16);
  dim3 grid((roi + block.x - 1) / block.x, (roi + block.y - 1) / block.y);
  k_bgra_to_f32<<<grid, block, 0, stream>>>(tex, roi, dstF32);

  cudaDestroyTextureObject(tex);
}
