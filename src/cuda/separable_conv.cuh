#ifndef __SEPARABLE_CONV_CUH__
#define __SEPARABLE_CONV_CUH__

#include <assert.h>
#include <cooperative_groups.h>
#include <helper_cuda.h>

__global__ void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW,
                                      int imageH, int pitch, float *d_kernel,
                                      int kl_radius, int halo_steps_);
__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW,
                                         int imageH, int pitch, float *d_kernel,
                                         int kl_radius, int halo_steps_);

class SeparableConv {
public:
  int kl_len;
  int kl_radius;
  int imageW;
  int imageH;
  int b_dim_x;
  int b_dim_y;
  int result_steps;
  int halo_steps;

  SeparableConv(float *h_kernel, int kernel_len_, int imageW_, int imageH_);
  SeparableConv() {}
  void init(float *h_kernel, int kernel_len_, int imageW_, int imageH_);
  ~SeparableConv();
  void conv(float *d_Dst, float *d_Src);

private:
  float *d_kernel;
  float *d_buff;
  size_t memsize;
};

#endif // __SEPARABLE_CONV_CUH__
