/**
 * name:       separable_conv.cu
 * usage:      --
 * author:     Ruicheng
 * date:       2021-09-23 09:45:56
 * version:    1.0
 * Env.:       CUDA 10.2, WIN 10
 */

#include <cooperative_groups.h>

#include "separable_conv.cuh"
namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW,
                                      int imageH, int pitch, float *d_kernel,
                                      int kl_radius, int halo_steps_) {

  int blockdim_x = blockDim.x;
  int blockdim_y = blockDim.y;
  int result_steps = kl_radius;
  int halo_steps = halo_steps_;

  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  extern __shared__ float s_Data[];
  int sidx;
  int n_s_row = (result_steps + 2 * halo_steps) * blockdim_x;
  // __shared__ float
  //     s_Data[blockdim_y][(result_steps + 2 * halo_steps) * blockdim_x];

  // Offset to the left halo edge
  const int baseX =
      (blockIdx.x * result_steps - halo_steps) * blockdim_x + threadIdx.x;
  const int baseY = blockIdx.y * blockdim_y + threadIdx.y;

  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

  // Load main data
#pragma unroll

  for (int i = halo_steps; i < halo_steps + result_steps; i++) {
    sidx = (threadIdx.y * n_s_row) + (threadIdx.x + i * blockdim_x);
    s_Data[sidx] = d_Src[i * blockdim_x];
  }

  // Load left halo
#pragma unroll

  for (int i = 0; i < halo_steps; i++) {
    sidx = (threadIdx.y * n_s_row) + (threadIdx.x + i * blockdim_x);
    s_Data[sidx] = (baseX >= -i * blockdim_x) ? d_Src[i * blockdim_x] : 0;
  }

  // Load right halo
#pragma unroll

  for (int i = halo_steps + result_steps;
       i < halo_steps + result_steps + halo_steps; i++) {
    sidx = (threadIdx.y * n_s_row) + (threadIdx.x + i * blockdim_x);
    s_Data[sidx] =
        (imageW - baseX > i * blockdim_x) ? d_Src[i * blockdim_x] : 0;
  }

  __syncthreads();

  // Compute and store results
  cg::sync(cta);
#pragma unroll

  for (int i = halo_steps; i < halo_steps + result_steps; i++) {
    float sum = 0;

#pragma unroll

    for (int j = -kl_radius; j <= kl_radius; j++) {
      sidx = (threadIdx.y * n_s_row) + (threadIdx.x + i * blockdim_x);
      sum += d_kernel[kl_radius - j] * s_Data[sidx + j];
    }

    d_Dst[i * blockdim_x] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW,
                                         int imageH, int pitch, float *d_kernel,
                                         int kl_radius, int halo_steps_) {

  int blockdim_x = blockDim.x;
  int blockdim_y = blockDim.y;
  int result_steps = kl_radius;
  int halo_steps = halo_steps_;

  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  extern __shared__ float s_Data[];
  int sidx;
  int n_s_row = (result_steps + 2 * halo_steps) * blockdim_y + 1;
  // __shared__
  //     float s_Data[blockdim_x][(result_steps + 2 * halo_steps) * blockdim_y +
  //     1];

  // Offset to the upper halo edge
  const int baseX = blockIdx.x * blockdim_x + threadIdx.x;
  const int baseY =
      (blockIdx.y * result_steps - halo_steps) * blockdim_y + threadIdx.y;
  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

  // Main data
#pragma unroll

  for (int i = halo_steps; i < halo_steps + result_steps; i++) {
    sidx = threadIdx.x * n_s_row + (threadIdx.y + i * blockdim_y);
    s_Data[sidx] = d_Src[i * blockdim_y * pitch];
  }

  // Upper halo
#pragma unroll

  for (int i = 0; i < halo_steps; i++) {
    sidx = threadIdx.x * n_s_row + (threadIdx.y + i * blockdim_y);
    s_Data[sidx] =
        (baseY >= -i * blockdim_y) ? d_Src[i * blockdim_y * pitch] : 0;
  }

  // Lower halo
#pragma unroll

  for (int i = halo_steps + result_steps;
       i < halo_steps + result_steps + halo_steps; i++) {
    sidx = threadIdx.x * n_s_row + (threadIdx.y + i * blockdim_y);
    s_Data[sidx] =
        (imageH - baseY > i * blockdim_y) ? d_Src[i * blockdim_y * pitch] : 0;
  }

  __syncthreads();

  // Compute and store results
  cg::sync(cta);
#pragma unroll

  for (int i = halo_steps; i < halo_steps + result_steps; i++) {
    float sum = 0;
#pragma unroll

    for (int j = -kl_radius; j <= kl_radius; j++) {
      sidx = threadIdx.x * n_s_row + (threadIdx.y + i * blockdim_y);
      sum += d_kernel[kl_radius - j] * s_Data[sidx + j];
    }

    d_Dst[i * blockdim_y * pitch] = sum;
  }
}

SeparableConv::SeparableConv(float *h_kernel, int kernel_len_, int imageW_,
                             int imageH_) {
  init(h_kernel, kernel_len_, imageW_, imageH_);
}

void SeparableConv::init(float *h_kernel, int kernel_len_, int imageW_,
                         int imageH_) {
  // kernel size
  kl_len = kernel_len_;
  kl_radius = kl_len / 2;
  imageW = imageW_;
  imageH = imageH_;
  memsize = imageW * imageH * sizeof(float);
  b_dim_x = 16;
  b_dim_y = 16;
  result_steps = kl_radius;
  halo_steps = 1;

  // set convolution kernel to device memory
  checkCudaErrors(cudaMalloc((void **)&d_kernel, kl_len * sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_kernel, h_kernel, kl_len * sizeof(float),
                             cudaMemcpyHostToDevice));

  // image buff
  checkCudaErrors(cudaMalloc((void **)&d_buff, memsize));
}

SeparableConv::~SeparableConv() {
  checkCudaErrors(cudaFree(d_kernel));
  checkCudaErrors(cudaFree(d_buff));
}

void SeparableConv::conv(float *d_Dst, float *d_Src) {
  // row
  assert(b_dim_x * halo_steps >= kl_radius);
  assert(imageW % (result_steps * b_dim_x) == 0);
  assert(imageH % b_dim_y == 0);

  dim3 blocks1(imageW / (result_steps * b_dim_x), imageH / b_dim_y);
  dim3 threads1(b_dim_x, b_dim_y);
  size_t share_size1 =
      b_dim_y * ((result_steps + 2 * halo_steps) * b_dim_x) * sizeof(float);

  convolutionRowsKernel<<<blocks1, threads1, share_size1>>>(
      d_buff, d_Src, imageW, imageH, imageW, d_kernel, kl_radius, halo_steps);
  getLastCudaError("convolutionRowsKernel() execution failed\n");

  // column
  assert(b_dim_y * halo_steps >= kl_radius);
  assert(imageW % b_dim_x == 0);
  assert(imageH % (result_steps * b_dim_y) == 0);

  dim3 blocks2(imageW / b_dim_x, imageH / (result_steps * b_dim_y));
  dim3 threads2(b_dim_x, b_dim_y);
  size_t share_size2 =
      b_dim_x * ((result_steps + 2 * halo_steps) * b_dim_y + 1) * sizeof(float);

  convolutionColumnsKernel<<<blocks2, threads2, share_size2>>>(
      d_Dst, d_buff, imageW, imageH, imageW, d_kernel, kl_radius, halo_steps);
  getLastCudaError("convolutionColumnsKernel() execution failed\n");

  checkCudaErrors(cudaDeviceSynchronize());
}
