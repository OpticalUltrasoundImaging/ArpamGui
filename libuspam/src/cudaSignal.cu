/**
Everything here assumes column major for 2D arrays
*/
#include "uspam/cudaSignal.cuh"
#include "uspam/cudaUtil.cuh"

#include <array>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cuda/std/cmath>

__global__ void kernelHilbert_r2c_freq_switch(cufftDoubleComplex *data,
                                              int rows, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  const int j = blockIdx.y * blockDim.y + threadIdx.y; // col
  if (i < rows && i < cols) {
    const int idx = j * rows + i;
    // data[idx] *= -1j;
    data[idx] = cuCmul(data[idx], make_cuDoubleComplex(0, -1));
  }
}

__global__ void kernelHilbert_r2c_scale_and_abs(double *real, double *imag,
                                                int rows, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) {
    const int idx = j * rows + i;
    real[idx] = cuCabs(
        make_cuDoubleComplex(real[idx], imag[idx] / static_cast<double>(rows)));
  }
}

void hilbert2(const double *x, double *out, const int rows, const int cols) {
  cufftHandle planr2c;
  cufftHandle planc2r;
  cudaStream_t stream = nullptr;

  std::array<int, 1> fft_size = {rows};
  const int dist = rows;
  const int stride = 1;

  CUFFT_CALL(cufftCreate(&planr2c));
  CUFFT_CALL(cufftCreate(&planc2r));
  // clang-format off
  CUFFT_CALL(cufftPlanMany(&planr2c, fft_size.size(), fft_size.data(),
                           nullptr, stride, dist,
                           nullptr, stride, dist,
                           CUFFT_D2Z, cols));

  CUFFT_CALL(cufftPlanMany(&planc2r, fft_size.size(), fft_size.data(),
                           nullptr, stride, dist,
                           nullptr, stride, dist,
                           CUFFT_Z2D, cols));
  // clang-format on
  CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUFFT_CALL(cufftSetStream(planr2c, stream));
  CUFFT_CALL(cufftSetStream(planc2r, stream));

  // Create device arrays
  cufftDoubleComplex *d_cx;
  double *d_real;
  double *d_imag;

  CUDA_RT_CALL(cudaMalloc(&d_cx, rows * cols * sizeof(cufftDoubleComplex)));
  CUDA_RT_CALL(cudaMalloc(&d_real, rows * cols * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc(&d_imag, rows * cols * sizeof(double)));

  CUDA_RT_CALL(cudaMemcpyAsync(d_real, x, rows * cols * sizeof(double),
                               cudaMemcpyHostToDevice));

  // Forward fft
  CUFFT_CALL(cufftExecD2Z(planr2c, d_real, d_cx));

  // Manipulate spectrum for Hilbert transform
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernelHilbert_r2c_freq_switch<<<numBlocks, threadsPerBlock>>>(d_cx, rows,
                                                                cols);

  // Backward fft
  CUFFT_CALL(cufftExecZ2D(planc2r, d_cx, d_imag));

  kernelHilbert_r2c_scale_and_abs<<<numBlocks, threadsPerBlock>>>(
      d_real, d_imag, rows, cols);

  // Copy to output
  CUDA_RT_CALL(cudaMemcpyAsync(out, d_real, rows * cols * sizeof(double),
                               cudaMemcpyDeviceToHost));

  // CUDA_RT_CALL(cudaMemcpy(out, d_imag, rows * cols * sizeof(double),
  //                         cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  // Cleanup
  CUDA_RT_CALL(cudaFree(d_cx));
  CUDA_RT_CALL(cudaFree(d_real));
  CUDA_RT_CALL(cudaFree(d_imag));

  CUDA_RT_CALL(cufftDestroy(planr2c));
  CUDA_RT_CALL(cufftDestroy(planc2r));

  CUDA_RT_CALL(cudaStreamDestroy(stream));

}

// void hilbert2_v2(const double *x, double *out, const int rows,
//                        const int cols) {
//
//   // Create cuFFT plans
//   cufftHandle plan_f, plan_b;
//   cudaStream_t stream = nullptr;
//
//   CUFFT_CALL(cufftCreate(&plan_b));
//   CUFFT_CALL(cufftCreate(&plan_f));
//
//   std::array<int, 1> fft_size{1};
//   const int dist = rows;
//   const int stride = 1;
//
//   // clang-format off
//   CUFFT_CALL(cufftPlanMany(&plan_f, fft_size.size(), fft_size.data(),
//                            nullptr, stride, dist,
//                            nullptr, stride, dist,
//                            CUFFT_D2Z, cols));
//
//   CUFFT_CALL(cufftPlanMany(&plan_b, fft_size.size(), fft_size.data(),
//                            nullptr, stride, dist,
//                            nullptr, stride, dist,
//                            CUFFT_Z2D, cols));
//   // clang-format on
//
//   CUFFT_CALL(cufftSetStream(plan_f, stream));
//   CUFFT_CALL(cufftSetStream(plan_b, stream));
//
//   // Create device arrays
//   // (inplace transform)
//   using Real = double;
//   using Complex = std::complex<Real>;
//   cufftDoubleComplex *d_cx;
//
//   CUDA_RT_CALL(cudaMalloc(&d_cx, rows * cols * sizeof(cufftDoubleComplex)));
//   CUDA_RT_CALL(cudaMemcpyAsync(d_real, x, rows * cols * sizeof(double),
//                                cudaMemcpyHostToDevice));
//
//   // Forward fft
//   CUFFT_CALL(cufftExecD2Z(plan_f, d_real, d_cx));
//
//   // Manipulate spectrum for Hilbert transform
//   dim3 threadsPerBlock(16, 16);
//   dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                  (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
//   kernelHilbert2<<<numBlocks, threadsPerBlock>>>(d_cx, rows, cols);
//
//   // Backward fft
//   CUFFT_ERROR(cufftExecZ2D(plan_b, d_cx, d_real));
//
//   const double fct = 1. / rows;
//   kernelAbs2<<<numBlocks, threadsPerBlock>>>(d_cx, d_real, rows, cols);
//
//   // Copy to output
//   CUDA_RT_CALL(cudaMemcpy(out, d_real, rows * cols * sizeof(double),
//                           cudaMemcpyDeviceToHost));
//
//   cudaDeviceSynchronize();
//
//   // Cleanup
//   CUDA_RT_CALL(cudaFree(d_cx));
//   CUDA_RT_CALL(cudaFree(d_real));
//   CUFFT_CALL(cufftDestroy(plan_f));
//   CUFFT_CALL(cufftDestroy(plan_b));
// }