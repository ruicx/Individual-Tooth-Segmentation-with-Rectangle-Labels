#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#define USE_SYCN 0

#if USE_SYCN
#define DEVICE_SYNC checkCudaErrors(cudaDeviceSynchronize());
#define DEVICE_SYNC_INFO printf("Using device synchronize.\n");
#else
#define DEVICE_SYNC
#define DEVICE_SYNC_INFO printf("Without device synchronize.\n");
#endif

#include "convolutionSeparable_common.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "separable_conv.cuh"
#include "utils.hpp"

// gradient
__global__ void gradient_kernel(float *d_grad_x, float *d_grad_y,
                                const float *d_Src, const int imageW,
                                const int imageH);

__global__ void gradient_edge_kernel(float *d_grad_x, float *d_grad_y,
                                     const float *d_Src, const int imageW,
                                     const int imageH);

extern "C" void gradientGPU(float *d_grad_x, float *d_grad_y,
                            const float *d_Src, const int imageW,
                            const int imageH);

// laplacian
__global__ void laplacian_kernel(float *d_Dst, float *d_buff1, float *d_buff2,
                                 const float *d_Src, const int imageW,
                                 const int imageH);

extern "C" void laplacianGPU(float *d_Dst, const float *d_Src, const int imageW,
                             const int imageH);

// Neumann Boundary Condition
__global__ void NeumannBC_kernel(float *d_ls, const int imageW,
                                 const int imageH);

void NeumannBC_GPU(float *d_ls, const int imageW, const int imageH);

// norm gradient
__global__ void norm_matrix_kernel(float *d_n_x, float *d_n_y,
                                   const float *d_grad_x, const float *d_grad_y,
                                   const int max_len);

// matrix operation
__global__ void matrix_sum_kernel(float *d_Dst, const float *d_mat1,
                                  const float *d_mat2, const int max_len);

__global__ void matrix_div_kernel(float *d_Dst, const float *d_mat1,
                                  const float *d_mat2, const int max_len);

__global__ void matrix_init_kernel(float *d_Dst, const float num,
                                   const int max_len);

__global__ void matrix_dirac_delta_kernel(float *d_Dst, const float *d_Src,
                                          const float epsilon,
                                          const int max_len);

//  LBF
__global__ void LBF_before_kernel(float *d_Hu, float *d_HuI, const float *d_ls,
                                  const float *d_img, float epsilon,
                                  const int max_len);

__global__ void LBF_after_kernel(float *d_f1, float *d_f2, const float *d_c1,
                                 const float *d_c2, const float *d_KI,
                                 const float *d_KONE, const int max_len);

__global__ void data_term_kernel(float *d_s1, float *d_s2, const float *d_f1,
                                 const float *d_f2, const float lambda1,
                                 const float lambda2, const int max_len);

__global__ void data_force_kernel(float *d_dataForce, const float *d_s1,
                                  const float *d_s2, const float *d_img,
                                  const float *d_KONE, const float lambda1,
                                  const float lambda2, const int max_len);

__global__ void iter_gradient_kernel(float *d_ls, const float *d_laplace,
                                     const float *d_K, const float *d_DrcU,
                                     const float *d_dataForce, const float nu,
                                     const float mu, const float time_step,
                                     const int max_len);

__global__ void fetch_mask_kernel(float *d_contour, const float *d_ls,
                                  const float level, const int max_len);

__global__ void clf_kernel(float *d_ls, const float *d_laplace,
                           const float *d_K, const float *d_DrcU,
                           const float *d_Hu, const float *d_dataForce,
                           const float *d_ccst, const float *d_pcst,
                           const float nu, const float mu, const float eta,
                           const float tau, const float time_step,
                           const int max_len);

__device__ float dirac_delta(float x, float epsilon);

__device__ float heaviside(float x, float epsilon);

//    RSF
class RSF {
public:
  float epsilon;
  float mu;
  float nu;
  float nu_show;
  int imageW;
  int imageH;
  int kernel_len;
  float lambda1;
  float lambda2;
  float time_step;
  float sigma;
  dim3 threads1d;
  dim3 blocks1d;
  dim3 threads2d;
  dim3 blocks2d;

  RSF(const float *h_Input, const float *h_ls, float *h_Kernel, int imageH_,
      int imageW_, int kernel_len_, float lambda1_, float lambda2_, float mu_,
      float epsilon_, float nu_, float time_step_);
  RSF(const float *h_Input, const float *h_ls, int imageH_, int imageW_,
      float sigma_, float lambda1_, float lambda2_, float mu_, float epsilon_,
      float nu_, float time_step_);
  RSF();
  ~RSF();
  virtual void hello(void);
  int localBinaryFit(void);
  int curvature_central_gpu(void);
  int get_DataForce(void);
  virtual int iter(void);
  int iter(int num);
  int fetch_result(float *h_out);
  int reinit(float *h_ls);
  int fetch_mask(float *h_out, float level);
  int fetch_contour(float *h_out, float level);
  int step_test(void);

protected:
  float *d_ls;
  float *d_img;
  float *d_Hu;
  float *d_HuI;
  float *d_KI;
  float *d_DrcU; // DrcU
  float *d_f1;
  float *d_f2;
  float *d_c1;
  float *d_c2;
  float *d_kl;   // decomposed convolution kernel
  float *d_nll;  // buffer
  float *d_nll2; // buffer
  float *d_KONE;
  float *d_Ksigma;
  float *d_ux;
  float *d_uy;
  float *d_nx;
  float *d_ny;
  float *d_K;
  float *d_s1;
  float *d_s2;
  float *d_dataForce;
  float *d_laplace;
  float *d_mask;
  float *d_contour;

  size_t memsize;
  int max_len;

  SeparableConv sep_conv;

  void set_memory(const float *h_Input, const float *h_ls, float *h_Kernel,
                  int imageH_, int imageW_, int kernel_len_, float lambda1_,
                  float lambda2_, float mu_, float epsilon_, float nu_,
                  float time_step_);
};

class CLF : public RSF {
public:
  float eta;
  float tau;

  CLF::CLF(const float *h_Input, const float *h_ls, const float *h_ccst,
           const float *h_pcst, int imageH_, int imageW_, float sigma_,
           float lambda1_, float lambda2_, float mu_, float epsilon_, float nu_,
           float eta_, float tau_, float time_step_);
  ~CLF();
  void hello(void);
  int iter(void);

protected:
  float *d_ccst; // context constract
  float *d_pcst; // position constract
};

#endif // __GRADIENT_H__
