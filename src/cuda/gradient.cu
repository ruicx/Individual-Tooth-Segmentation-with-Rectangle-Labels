/**
 * name:       gradient.cu
 * usage:      --
 * author:     Ruicheng
 * date:       2021-08-12 16:59:07
 * version:    1.0
 * Env.:       CUDA 10.2, WIN 10
 */

#include "gradient.cuh"

// ██╗  ██╗    ███████╗    ██████╗     ███╗   ██╗    ███████╗    ██╗
// ██║ ██╔╝    ██╔════╝    ██╔══██╗    ████╗  ██║    ██╔════╝    ██║
// █████╔╝     █████╗      ██████╔╝    ██╔██╗ ██║    █████╗      ██║
// ██╔═██╗     ██╔══╝      ██╔══██╗    ██║╚██╗██║    ██╔══╝      ██║
// ██║  ██╗    ███████╗    ██║  ██║    ██║ ╚████║    ███████╗    ███████╗
// ╚═╝  ╚═╝    ╚══════╝    ╚═╝  ╚═╝    ╚═╝  ╚═══╝    ╚══════╝    ╚══════╝
//

// gradient
__global__ void gradient_kernel(float *d_grad_x, float *d_grad_y,
                                const float *d_Src, const int imageW,
                                const int imageH) {
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int ix = blockIdx.x * blockDim.x + threadIdx.x;

  if (ix > 0 && ix < imageW - 1) {
    int idx = iy * imageW + ix;
    d_grad_x[idx] = (d_Src[idx + 1] - d_Src[idx - 1]) / 2;
  }

  if (iy > 0 && iy < imageH - 1) {
    int idx = iy * imageW + ix;
    d_grad_y[idx] = (d_Src[idx + imageW] - d_Src[idx - imageW]) / 2;
  }
}

__global__ void gradient_edge_kernel(float *d_grad_x, float *d_grad_y,
                                     const float *d_Src, const int imageW,
                                     const int imageH) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < imageH) {
    d_grad_x[ix * imageW] = d_Src[ix * imageW + 1] - d_Src[ix * imageW];
    d_grad_x[(ix + 1) * imageW - 1] =
        d_Src[(ix + 1) * imageW - 1] - d_Src[(ix + 1) * imageW - 2];
  }
  if (ix < imageW) {
    d_grad_y[ix] = d_Src[ix + imageW] - d_Src[ix];
    d_grad_y[(imageH - 1) * imageW + ix] =
        d_Src[(imageH - 1) * imageW + ix] - d_Src[(imageH - 2) * imageW + ix];
  }
}

extern "C" void gradientGPU(float *d_grad_x, float *d_grad_y,
                            const float *d_Src, const int imageW,
                            const int imageH) {

  dim3 threads(32, 32);
  dim3 blocks(imageW / threads.x, imageH / threads.y);

  gradient_kernel<<<blocks, threads>>>(d_grad_x, d_grad_y, d_Src, imageW,
                                       imageH);
  getLastCudaError("gradient_kernel() execution failed\n");

  dim3 threadE(32);
  dim3 blockE(imageW / threadE.x + 1);

  gradient_edge_kernel<<<blockE, threadE>>>(d_grad_x, d_grad_y, d_Src, imageW,
                                            imageH);
  getLastCudaError("gradient_edge_kernel() execution failed\n");
}

// laplacian
__global__ void laplacian_kernel(float *d_Dst, float *d_buff1, float *d_buff2,
                                 const float *d_Src, const int imageW,
                                 const int imageH) {
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = iy * imageW + ix;
  float up, down, left, right, mid;

  // horizontal
  if (ix > 0 && ix < imageW - 1) {
    mid = d_Src[idx];
    left = d_Src[idx - 1];
    right = d_Src[idx + 1];
    d_buff1[idx] = (left + right) / 2.0 - mid;
  }
  __syncthreads();
  if (ix == 0) {
    d_buff1[idx] = d_buff1[idx + 1] * 2.0 - d_buff1[idx + 2];
  } else if (ix == (imageW - 1)) {
    d_buff1[idx] = d_buff1[idx - 1] * 2.0 - d_buff1[idx - 2];
  }
  __syncthreads();

  // vertical
  if (iy > 0 && iy < imageH - 1) {
    mid = d_Src[idx];
    up = d_Src[idx - imageW];
    down = d_Src[idx + imageW];
    d_buff2[idx] = (up + down) / 2.0 - mid;
  }
  __syncthreads();
  if (iy == 0) {
    d_buff2[idx] = d_buff2[idx + imageW] * 2.0 - d_buff2[idx + 2 * imageW];
  } else if (iy == (imageH - 1)) {
    d_buff2[idx] = d_buff2[idx - imageW] * 2.0 - d_buff2[idx - 2 * imageW];
  }
  __syncthreads();

  // result
  d_Dst[idx] = (d_buff1[idx] + d_buff2[idx]) / 2.0;
}

extern "C" void laplacianGPU(float *d_Dst, const float *d_Src, const int imageW,
                             const int imageH) {

  float *d_buff1, *d_buff2;
  checkCudaErrors(
      cudaMalloc((void **)&d_buff1, imageW * imageH * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_buff2, imageW * imageH * sizeof(float)));
  dim3 threads(32, 32);
  dim3 blocks(imageW / threads.x, imageH / threads.y);

  laplacian_kernel<<<blocks, threads>>>(d_Dst, d_buff1, d_buff2, d_Src, imageW,
                                        imageH);

  getLastCudaError("laplacian_kernel() execution failed\n");
  checkCudaErrors(cudaFree(d_buff1));
  checkCudaErrors(cudaFree(d_buff2));
}

// Neumann Boundary Condition
__global__ void NeumannBC_kernel(float *d_ls, const int imageW,
                                 const int imageH) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;

  // row
  if (ix > 0 && ix < imageW - 1) {
    d_ls[ix] = d_ls[ix + 2 * imageW];
    float *last_row = d_ls + (imageH - 1) * imageW;
    last_row[ix] = last_row[ix - 2 * imageW];
  }

  // column
  if (ix > 0 && ix < imageH - 1) {
    d_ls[ix * imageW] = d_ls[2 + ix * imageW];
    float *last_col = d_ls + ix * imageW + imageW - 1;
    last_col[0] = last_col[-2];
  }

  // corner
  if (ix == 0) {
    float *last_row = d_ls + imageW * (imageH - 1);
    d_ls[0] = d_ls[2 + 2 * imageW];               // up left
    d_ls[imageW - 1] = d_ls[3 * imageW - 3];      // up right
    last_row[0] = last_row[2 - 2 * imageW];       // low left
    last_row[imageW - 1] = last_row[-imageW - 3]; // low right
  }
}

void NeumannBC_GPU(float *d_ls, const int imageW, const int imageH) {

  // edge
  dim3 threadE(128);
  dim3 blockE(imageW / threadE.x + 1);
  NeumannBC_kernel<<<blockE, threadE>>>(d_ls, imageW, imageH);
  getLastCudaError("NeumannBC_kernel() execution failed\n");
  checkCudaErrors(cudaDeviceSynchronize());
}

// norm gradient
__global__ void norm_matrix_kernel(float *d_n_x, float *d_n_y,
                                   const float *d_grad_x, const float *d_grad_y,
                                   const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    float norm = sqrtf(d_grad_x[idx] * d_grad_x[idx] +
                       d_grad_y[idx] * d_grad_y[idx] + 1e-10);
    d_n_x[idx] = d_grad_x[idx] / norm;
    d_n_y[idx] = d_grad_y[idx] / norm;
  }
}

// matrix operation
__global__ void matrix_sum_kernel(float *d_Dst, const float *d_mat1,
                                  const float *d_mat2, const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    d_Dst[idx] = d_mat1[idx] + d_mat2[idx];
  }
}

__global__ void matrix_div_kernel(float *d_Dst, const float *d_mat1,
                                  const float *d_mat2, const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    d_Dst[idx] = d_mat1[idx] / (d_mat2[idx] + 1e-6);
  }
}

__global__ void matrix_init_kernel(float *d_Dst, const float num,
                                   const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    d_Dst[idx] = num;
  }
}

__global__ void matrix_dirac_delta_kernel(float *d_Dst, const float *d_Src,
                                          const float epsilon,
                                          const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    d_Dst[idx] = dirac_delta(d_Src[idx], epsilon);
  }
}

__device__ float dirac_delta(float x, float epsilon) {
  return (epsilon / 3.14159265358979) / (epsilon * epsilon + x * x);
}

__device__ float heaviside(float x, float epsilon) {
  return 0.5 * (1 + (2 / 3.14159265358979) * atanf(x / epsilon));
}

// local binary fit
__global__ void LBF_before_kernel(float *d_Hu, float *d_HuI, const float *d_ls,
                                  const float *d_img, float epsilon,
                                  const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    float hu = heaviside(d_ls[idx], epsilon);
    d_Hu[idx] = hu;
    d_HuI[idx] = hu * d_img[idx];
  }
}

__global__ void LBF_after_kernel(float *d_f1, float *d_f2, const float *d_c1,
                                 const float *d_c2, const float *d_KI,
                                 const float *d_KONE, const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    d_f1[idx] = d_c2[idx] / (d_c1[idx]);
    d_f2[idx] = (d_KI[idx] - d_c2[idx]) / (d_KONE[idx] - d_c1[idx]);
  }
}

__global__ void data_term_kernel(float *d_s1, float *d_s2, const float *d_f1,
                                 const float *d_f2, const float lambda1,
                                 const float lambda2, const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    d_s1[idx] = (lambda1 * (d_f1[idx] * d_f1[idx])) -
                (lambda2 * (d_f2[idx] * d_f2[idx]));
    d_s2[idx] = lambda1 * d_f1[idx] - lambda2 * d_f2[idx];
  }
}

__global__ void data_force_kernel(float *d_dataForce, const float *d_s1,
                                  const float *d_s2, const float *d_img,
                                  const float *d_KONE, const float lambda1,
                                  const float lambda2, const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    d_dataForce[idx] =
        (lambda1 - lambda2) * d_KONE[idx] * d_img[idx] * d_img[idx] +
        d_s1[idx] - 2 * d_img[idx] * d_s2[idx];
  }
}

__global__ void iter_gradient_kernel(float *d_ls, const float *d_laplace,
                                     const float *d_K, const float *d_DrcU,
                                     const float *d_dataForce, const float nu,
                                     const float mu, const float time_step,
                                     const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float A, P, L;
  if (idx < max_len) {
    A = -d_DrcU[idx] * d_dataForce[idx];
    P = mu * (4 * d_laplace[idx] - d_K[idx]);
    L = nu * d_DrcU[idx] * d_K[idx];

    d_ls[idx] = d_ls[idx] + time_step * (A + P + L);
  }
}

__global__ void fetch_mask_kernel(float *d_contour, const float *d_ls,
                                  const float level, const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_len) {
    // d_contour[idx] = dirac_delta(d_ls[idx], 1) > 0.02? 1 : 0;
    d_contour[idx] = d_ls[idx] > level ? 1 : 0;
  }
}

// ██████╗     ███████╗    ███████╗
// ██╔══██╗    ██╔════╝    ██╔════╝
// ██████╔╝    ███████╗    █████╗
// ██╔══██╗    ╚════██║    ██╔══╝
// ██║  ██║    ███████║    ██║
// ╚═╝  ╚═╝    ╚══════╝    ╚═╝

RSF::RSF(const float *h_Input, const float *h_ls, float *h_Kernel, int imageH_,
         int imageW_, int kernel_len_, float lambda1_, float lambda2_,
         float mu_, float epsilon_, float nu_, float time_step_) {

  // set rsf memory
  set_memory(h_Input, h_ls, h_Kernel, imageH_, imageW_, kernel_len_, lambda1_,
             lambda2_, mu_, epsilon_, nu_, time_step_);
}

void RSF::set_memory(const float *h_Input, const float *h_ls, float *h_Kernel,
                     int imageH_, int imageW_, int kernel_len_, float lambda1_,
                     float lambda2_, float mu_, float epsilon_, float nu_,
                     float time_step_) {
  imageH = imageH_;
  imageW = imageW_;
  kernel_len = kernel_len_;
  lambda1 = lambda1_;
  lambda2 = lambda2_;
  mu = mu_;
  nu = nu_ * 255 * 255;
  nu_show = nu_;
  epsilon = epsilon_;
  time_step = time_step_;
  memsize = imageW * imageH * sizeof(float);
  max_len = imageW * imageH;
  printf("Load image in resolution (%d, %d)\n", imageH, imageW);
  printf("memmory size is %zd\n", memsize);

  // cuda graid 2d
  dim3 threads2d_(16, 16);
  dim3 blocks2d_(imageW / threads2d_.x, imageH / threads2d_.y);
  threads2d = threads2d_;
  blocks2d = blocks2d_;

  // cuda graid 1d
  dim3 threads1d_(128);
  dim3 blocks1d_(imageW * imageH / threads1d_.x);
  threads1d = threads1d_;
  blocks1d = blocks1d_;

  // convolution kernel
  // setConvolutionKernel(h_Kernel);

  // define convolution
  sep_conv.init(h_Kernel, kernel_len, imageW, imageH);

  // alloc cuda memory
  checkCudaErrors(cudaMalloc((void **)&d_ls, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_img, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_Hu, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_HuI, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_KI, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_f1, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_f2, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_c1, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_c2, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_kl, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_KONE, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_nll, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_nll2, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_DrcU, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_ux, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_uy, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_nx, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_ny, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_K, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_s1, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_s2, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_dataForce, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_laplace, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_mask, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_contour, memsize));

  // copy image and initial level set to cuda
  checkCudaErrors(cudaMemcpy(d_img, h_Input, memsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ls, h_ls, memsize, cudaMemcpyHostToDevice));

  // prepare d_KONE and d_KI
  // Matlab: KONE=conv2(ones(size(Img)),K,'same');
  matrix_init_kernel<<<blocks1d, threads1d>>>(d_nll, 1.0, max_len);
  sep_conv.conv(d_KONE, d_nll);

  // Matlab: KI=conv2(Img,K,'same')
  sep_conv.conv(d_KI, d_img);

  DEVICE_SYNC_INFO
}

RSF::RSF(const float *h_Input, const float *h_ls, int imageH_, int imageW_,
         float sigma_, float lambda1_, float lambda2_, float mu_,
         float epsilon_, float nu_, float time_step_) {
  //  get gaussian kernel
  sigma = sigma_;
  int kl_len = round(2 * sigma) * 2 + 1;
  float *h_Kernel = (float *)malloc(kl_len * sizeof(float));
  get_gaussian_kernel(h_Kernel, kl_len, sigma);

  // initial RSF
  set_memory(h_Input, h_ls, h_Kernel, imageH_, imageW_, kl_len, lambda1_,
             lambda2_, mu_, epsilon_, nu_, time_step_);

  free(h_Kernel);
}

RSF::RSF() {}

RSF::~RSF() {
  checkCudaErrors(cudaFree(d_ls));
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_Hu));
  checkCudaErrors(cudaFree(d_HuI));
  checkCudaErrors(cudaFree(d_KI));
  checkCudaErrors(cudaFree(d_f1));
  checkCudaErrors(cudaFree(d_f2));
  checkCudaErrors(cudaFree(d_c1));
  checkCudaErrors(cudaFree(d_c2));
  checkCudaErrors(cudaFree(d_kl));
  checkCudaErrors(cudaFree(d_KONE));
  checkCudaErrors(cudaFree(d_nll));
  checkCudaErrors(cudaFree(d_nll2));
  checkCudaErrors(cudaFree(d_DrcU));
  checkCudaErrors(cudaFree(d_ux));
  checkCudaErrors(cudaFree(d_uy));
  checkCudaErrors(cudaFree(d_nx));
  checkCudaErrors(cudaFree(d_ny));
  checkCudaErrors(cudaFree(d_K));
  checkCudaErrors(cudaFree(d_s1));
  checkCudaErrors(cudaFree(d_s2));
  checkCudaErrors(cudaFree(d_dataForce));
  checkCudaErrors(cudaFree(d_laplace));
  checkCudaErrors(cudaFree(d_mask));
  checkCudaErrors(cudaFree(d_contour));
}

void RSF::hello(void) {
  printf("RSF solver based on CUDA\n");
  printf("image size (W, H): (%d, %d)\n", imageW, imageH);
  printf("lambda1: %.6f\n", lambda1);
  printf("lambda2: %.6f\n", lambda2);
  printf("mu: %.6f\n", mu);
  printf("nu: %.6f * 255 * 255\n", nu_show);
  printf("epsilon: %.6f\n", epsilon);
  printf("time_step: %.6f\n", time_step);
}

// compute curvature
int RSF::curvature_central_gpu(void) {
  // gradient
  gradientGPU(d_ux, d_uy, d_ls, imageW, imageH);

  norm_matrix_kernel<<<blocks1d, threads1d>>>(d_nx, d_ny, d_ux, d_uy, max_len);
  getLastCudaError("norm_matrix_kernel() execution failed\n");

  // gradient
  gradientGPU(d_ux, d_nll, d_nx, imageW, imageH);
  gradientGPU(d_nll, d_uy, d_ny, imageW, imageH);

  // sum
  matrix_sum_kernel<<<blocks1d, threads1d>>>(d_K, d_ux, d_uy, max_len);
  getLastCudaError("matrix_sum_kernel() execution failed\n");
  DEVICE_SYNC

  return 0;
}

int RSF::localBinaryFit(void) {
  LBF_before_kernel<<<blocks1d, threads1d>>>(d_Hu, d_HuI, d_ls, d_img, epsilon,
                                             max_len);
  getLastCudaError("LBF_before_kernel() execution failed\n");
  DEVICE_SYNC

  // Matlab: c1=conv2(Hu,Ksigma,'same');
  sep_conv.conv(d_c1, d_Hu);

  // Matlab: c2=conv2(I,Ksigma,'same');
  sep_conv.conv(d_c2, d_HuI);

  LBF_after_kernel<<<blocks1d, threads1d>>>(d_f1, d_f2, d_c1, d_c2, d_KI,
                                            d_KONE, max_len);
  getLastCudaError("LBF_after_kernel() execution failed\n");
  DEVICE_SYNC
  return 0;
}

int RSF::get_DataForce(void) {
  data_term_kernel<<<blocks1d, threads1d>>>(d_s1, d_s2, d_f1, d_f2, lambda1,
                                            lambda2, max_len);
  getLastCudaError("data_term_kernel() execution failed\n");
  DEVICE_SYNC

  sep_conv.conv(d_s1, d_s1);
  sep_conv.conv(d_s2, d_s2);

  data_force_kernel<<<blocks1d, threads1d>>>(d_dataForce, d_s1, d_s2, d_img,
                                             d_KONE, lambda1, lambda2, max_len);
  getLastCudaError("data_force_kernel() execution failed\n");
  DEVICE_SYNC

  return 0;
}

int RSF::iter(void) {
  NeumannBC_GPU(d_ls, imageW, imageH);
  curvature_central_gpu();

  matrix_dirac_delta_kernel<<<blocks1d, threads1d>>>(d_DrcU, d_ls, epsilon,
                                                     max_len);
  getLastCudaError("matrix_dirac_delta_kernel() execution failed\n");
  DEVICE_SYNC

  localBinaryFit();
  get_DataForce();

  laplacian_kernel<<<blocks2d, threads2d>>>(d_laplace, d_nll, d_nll2, d_ls,
                                            imageW, imageH);
  getLastCudaError("laplacian_kernel() execution failed\n");
  DEVICE_SYNC

  iter_gradient_kernel<<<blocks1d, threads1d>>>(
      d_ls, d_laplace, d_K, d_DrcU, d_dataForce, nu, mu, time_step, max_len);
  getLastCudaError("iter_gradient_kernel() execution failed\n");
  DEVICE_SYNC

  return 0;
}

int RSF::iter(int num) {
  // start timer
  auto start = std::chrono::system_clock::now();

  for (int aa = 0; aa < num; aa++) {
    // printf("RSF iternation %d\n", aa);
    iter();
  }

  // stop timer and count time cost
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  double iter_time = elapsed.count();

  printf("Iteration finished, Total time = %.5f s,"
         "Time = %.5f s/iter, Iteration = %d,"
         "Size = (%d, %d), Throughput = %.4f MPixels/sec \n",
         iter_time, iter_time / num, num, imageW, imageH,
         (1.0e-6 * (double)(imageW * imageH) / iter_time));

  return 0;
}

int RSF::fetch_result(float *h_out) {
  checkCudaErrors(cudaMemcpy(h_out, d_ls, memsize, cudaMemcpyDeviceToHost));
  return 0;
}

int RSF::fetch_mask(float *h_out, float level) {
  fetch_mask_kernel<<<blocks1d, threads1d>>>(d_mask, d_ls, level, max_len);
  getLastCudaError("fetch_mask_kernel() execution failed\n");
  DEVICE_SYNC
  checkCudaErrors(cudaMemcpy(h_out, d_mask, memsize, cudaMemcpyDeviceToHost));
  return 0;
}

int RSF::fetch_contour(float *h_out, float level) {
  // get mask
  fetch_mask_kernel<<<blocks1d, threads1d>>>(d_mask, d_ls, level, max_len);
  getLastCudaError("fetch_mask_kernel() execution failed\n");
  DEVICE_SYNC
  // get contour
  laplacian_kernel<<<blocks2d, threads2d>>>(d_contour, d_nll, d_nll2, d_mask,
                                            imageW, imageH);
  getLastCudaError("laplacian_kernel() execution failed\n");
  DEVICE_SYNC
  checkCudaErrors(
      cudaMemcpy(h_out, d_contour, memsize, cudaMemcpyDeviceToHost));
  return 0;
}

int RSF::reinit(float *h_ls) {
  checkCudaErrors(cudaMemcpy(d_ls, h_ls, memsize, cudaMemcpyHostToDevice));
  return 0;
}

int RSF::step_test(void) {
  using namespace std;

  double check_err;
  float *h_gt = (float *)malloc(memsize);
  float *h_iter = (float *)malloc(memsize);

  print_sep_line("RSF Step Test");

  // load initial level set
  load_binary(h_iter, max_len, "./image/val_ls.bin");
  checkCudaErrors(cudaMemcpy(d_ls, h_iter, memsize, cudaMemcpyHostToDevice));
  save_binary(h_iter, max_len, "./image/iter_ls.bin");

  // check Neumann boundary condiction
  NeumannBC_GPU(d_ls, imageW, imageH);
  load_binary(h_gt, max_len, "./image/val_ls_nb.bin");
  checkCudaErrors(cudaMemcpy(h_iter, d_ls, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_ls_nb.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3
      ? printf("NeumannBC_GPU Pass, err %.6lf\n", check_err)
      : printf(">>> NeumannBC_GPU Fail, err %.6lf <<<\n", check_err);

  // check curvature centure
  curvature_central_gpu();
  load_binary(h_gt, max_len, "./image/val_K_curvature.bin");
  checkCudaErrors(cudaMemcpy(h_iter, d_K, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_K_curvature.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("curvature Pass, err %.6lf\n", check_err)
                   : printf(">>> curvature Fail, err %.6lf <<<\n", check_err);

  // check DrcU
  matrix_dirac_delta_kernel<<<blocks1d, threads1d>>>(d_DrcU, d_ls, epsilon,
                                                     max_len);
  getLastCudaError("matrix_dirac_delta_kernel() execution failed\n");
  DEVICE_SYNC
  load_binary(h_gt, max_len, "./image/val_DrcU.bin");
  checkCudaErrors(cudaMemcpy(h_iter, d_DrcU, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_DrcU.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("DrcU Pass, err %.6lf\n", check_err)
                   : printf(">>> DrcU Fail, err %.6lf <<< \n", check_err);

  // check data force
  localBinaryFit();
  // check c1
  checkCudaErrors(cudaMemcpy(h_iter, d_c1, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_lbf_c1.bin");
  load_binary(h_gt, max_len, "./image/val_lbf_c1.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("c1 Test 1 Pass, err %.6f\n", check_err)
                   : printf(">>> c1 Test 1 Fail, err %.6f <<<\n", check_err);
  // check c2
  checkCudaErrors(cudaMemcpy(h_iter, d_c2, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_lbf_c2.bin");
  load_binary(h_gt, max_len, "./image/val_lbf_c2.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("c2 Test 1 Pass, err %.6f\n", check_err)
                   : printf(">>> c2 Test 1 Fail, err %.6f <<<\n", check_err);
  // check f1
  checkCudaErrors(cudaMemcpy(h_iter, d_f1, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_lbf_f1.bin");
  load_binary(h_gt, max_len, "./image/val_lbf_f1.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("f1 Test 1 Pass, err %.6f\n", check_err)
                   : printf(">>> f1 Test 1 Fail, err %.6f <<<\n", check_err);
  // check f2
  checkCudaErrors(cudaMemcpy(h_iter, d_f2, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_lbf_f2.bin");
  load_binary(h_gt, max_len, "./image/val_lbf_f2.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("f2 Test 1 Pass, err %.6f\n", check_err)
                   : printf(">>> f2 Test 1 Fail, err %.6f <<<\n", check_err);

  get_DataForce();
  // check cs1
  checkCudaErrors(cudaMemcpy(h_iter, d_s1, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_lbf_cs1.bin");
  load_binary(h_gt, max_len, "./image/val_lbf_cs1.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("cs1 Test 1 Pass, err %.6f\n", check_err)
                   : printf(">>> cs1 Test 1 Fail, err %.6f <<<\n", check_err);
  // check s2
  checkCudaErrors(cudaMemcpy(h_iter, d_s2, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_lbf_cs2.bin");
  load_binary(h_gt, max_len, "./image/val_lbf_cs2.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("cs2 Test 1 Pass, err %.6f\n", check_err)
                   : printf(">>> cs2 Test 1 Fail, err %.6f <<<\n", check_err);
  //  check dataForce
  load_binary(h_gt, max_len, "./image/val_dataForce.bin");
  checkCudaErrors(
      cudaMemcpy(h_iter, d_dataForce, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_dataForce.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("dataForce Pass, err %.6lf\n", check_err)
                   : printf(">>> dataForce Fail, err %.6lf <<<\n", check_err);

  laplacian_kernel<<<blocks2d, threads2d>>>(d_laplace, d_nll, d_nll2, d_ls,
                                            imageW, imageH);
  getLastCudaError("laplacian_kernel() execution failed\n");
  DEVICE_SYNC

  iter_gradient_kernel<<<blocks1d, threads1d>>>(
      d_ls, d_laplace, d_K, d_DrcU, d_dataForce, nu, mu, time_step, max_len);
  getLastCudaError("iter_gradient_kernel() execution failed\n");
  DEVICE_SYNC
  load_binary(h_gt, max_len, "./image/val_lbf_update_u.bin");
  checkCudaErrors(cudaMemcpy(h_iter, d_ls, memsize, cudaMemcpyDeviceToHost));
  save_binary(h_iter, max_len, "./image/iter_lbf_update_u.bin");
  check_err = check_binary(h_gt, h_iter, max_len);
  check_err < 1e-3 ? printf("gradient Pass, err %.6lf\n", check_err)
                   : printf(">>> gradient Fail, err %.6lf <<< \n", check_err);

  free(h_gt);
  free(h_iter);
  return 0;
}

//  ██████╗    ██╗         ███████╗
// ██╔════╝    ██║         ██╔════╝
// ██║         ██║         █████╗
// ██║         ██║         ██╔══╝
// ╚██████╗    ███████╗    ██║
//  ╚═════╝    ╚══════╝    ╚═╝
//

__global__ void clf_kernel(float *d_ls, const float *d_laplace,
                           const float *d_K, const float *d_DrcU,
                           const float *d_Hu, const float *d_dataForce,
                           const float *d_ccst, const float *d_pcst,
                           const float nu, const float mu, const float eta,
                           const float tau, const float time_step,
                           const int max_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float A, P, L, Ctx, Pst;
  if (idx < max_len) {
    A = -d_DrcU[idx] * d_dataForce[idx];
    P = mu * (4 * d_laplace[idx] - d_K[idx]);
    L = nu * d_DrcU[idx] * d_K[idx];
    Ctx = -eta * 2. * (d_Hu[idx] - d_ccst[idx]) * d_DrcU[idx];
    // Ctx = -eta * 2. * (heaviside(d_ls[idx], 1) - d_ccst[idx]) * d_DrcU[idx];
    // Ctx = eta * (d_ls[idx] - d_ccst[idx]);
    Pst = tau * d_DrcU[idx] * d_pcst[idx];

    d_ls[idx] = d_ls[idx] + time_step * (A + P + L - Ctx - Pst);
  }
}

CLF::CLF(const float *h_Input, const float *h_ls, const float *h_ccst,
         const float *h_pcst, int imageH_, int imageW_, float sigma_,
         float lambda1_, float lambda2_, float mu_, float epsilon_, float nu_,
         float eta_, float tau_, float time_step_) {
  sigma = sigma_;
  eta = eta_;
  tau = tau_;

  //  get gaussian kernel
  int kl_len = round(2 * sigma) * 2 + 1;
  float *h_Kernel = (float *)malloc(kl_len * sizeof(float));
  get_gaussian_kernel(h_Kernel, kl_len, sigma);

  // initial CLF
  set_memory(h_Input, h_ls, h_Kernel, imageH_, imageW_, kl_len, lambda1_,
             lambda2_, mu_, epsilon_, nu_, time_step_);

  // set memory for constract terms
  checkCudaErrors(cudaMalloc((void **)&d_ccst, memsize));
  checkCudaErrors(cudaMalloc((void **)&d_pcst, memsize));

  // copy constract terms prior to cuda
  checkCudaErrors(cudaMemcpy(d_ccst, h_ccst, memsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pcst, h_pcst, memsize, cudaMemcpyHostToDevice));

  // save_binary(h_ccst, imageW * imageH, "./image/iterimage_2048_contour.bin");

  free(h_Kernel);
}

CLF::~CLF() {
  checkCudaErrors(cudaFree(d_ccst));
  checkCudaErrors(cudaFree(d_pcst));
}

void CLF::hello(void) {
  printf("CLF solver based on CUDA\n");
  printf("image size (W, H): (%d, %d)\n", imageW, imageH);
  printf("lambda1: %.6f\n", lambda1);
  printf("lambda2: %.6f\n", lambda2);
  printf("mu: %.6f\n", mu);
  printf("nu: %.6f * 255 * 255\n", nu_show);
  printf("epsilon: %.6f\n", epsilon);
  printf("eta: %.6f\n", eta);
  printf("tau: %.6f\n", tau);
  printf("time_step: %.6f\n", time_step);
}

int CLF::iter(void) {
  // printf("CLF\n");
  NeumannBC_GPU(d_ls, imageW, imageH);
  curvature_central_gpu();

  matrix_dirac_delta_kernel<<<blocks1d, threads1d>>>(d_DrcU, d_ls, epsilon,
                                                     max_len);
  getLastCudaError("matrix_dirac_delta_kernel() execution failed\n");
  DEVICE_SYNC

  localBinaryFit();
  get_DataForce();

  laplacian_kernel<<<blocks2d, threads2d>>>(d_laplace, d_nll, d_nll2, d_ls,
                                            imageW, imageH);
  getLastCudaError("laplacian_kernel() execution failed\n");
  DEVICE_SYNC

  clf_kernel<<<blocks1d, threads1d>>>(d_ls, d_laplace, d_K, d_DrcU, d_Hu,
                                      d_dataForce, d_ccst, d_pcst, nu, mu, eta,
                                      tau, time_step, max_len);
  getLastCudaError("iter_gradient_kernel() execution failed\n");
  DEVICE_SYNC

  return 0;
}
