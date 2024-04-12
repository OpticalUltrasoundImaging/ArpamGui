#include "uspam/cudaSignal.cuh"
#include "uspam/cudaUtil.cuh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void kernelHilbert(cufftDoubleComplex *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    if (idx > 0 && idx < N / 2) {
      data[idx].x *= 2.0;
      data[idx].y *= 2.0;
    } else if (idx > N / 2) {
      data[idx].x = 0.0;
      data[idx].y = 0.0;
    }
  }
}

__global__ void kernelAbs(cufftDoubleComplex *data, double *real, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    real[i] = cuCabs(data[i]);
  }
}

void hilbertTransform(const double *input, double *output, int N) {
  cufftHandle plan_f;
  cufftDoubleComplex *data;
  double *real;

  // Allocate memory on device
  HANDLE_CUDA_ERROR(cudaMalloc(&data, N * sizeof(cufftDoubleComplex)));
  HANDLE_CUDA_ERROR(cudaMalloc(&real, N * sizeof(double)));

  // Create a batched 1D FFT plan
  HANDLE_CUFFT_ERROR(cufftPlan1d(&plan_f, N, CUFFT_R2C, 1));

  // Copy input data to device and pad the imaginary part with 0
  for (int i = 0; i < N; ++i) {
    HANDLE_CUDA_ERROR(cudaMemcpy(&data[i], &input[i], sizeof(double),
                                 cudaMemcpyHostToDevice));
    data[i].y = 0.0;
  }

  // Execute the FFT plan
  HANDLE_CUFFT_ERROR(
      cufftExecD2Z(plan_f, (double *)data, (cufftDoubleComplex *)data));

  // Manupulate the spectrum for Hilbert Transform
  // Kernel to set multipliers for Hilbert transform
  kernelHilbert<<<(N + 255) / 256, 256>>>(data, N);

  // Execute the inverse FFT
  HANDLE_CUFFT_ERROR(cufftExecZ2D(plan_f, (cufftDoubleComplex *)data,
                                  (cufftDoubleReal *)data));

  kernelAbs<<<(N + 255) / 256, 256>>>(data, real, N);

  // Copy the result back to host
  HANDLE_CUDA_ERROR(
      cudaMemcpy(output, real, N * sizeof(double), cudaMemcpyDeviceToHost));

  // Cleanup
  HANDLE_CUDA_ERROR(cudaFree(data));
  HANDLE_CUDA_ERROR(cudaFree(real));
  HANDLE_CUFFT_ERROR(cufftDestroy(plan_f));
}

__global__ void kernelHilbert2(cufftDoubleComplex *data, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) {
    if (i > 0 && i < rows / 2) {
      data[j * rows + i].x *= 2.0;
      data[j * rows + i].y *= 2.0;
    } else if (i > rows / 2) {
      data[j * rows + i].x = 0.0;
      data[j * rows + i].y = 0.0;
    }
  }
}

__global__ void kernelAbs2(cufftDoubleComplex *data, double *real, int rows,
                           int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) {
    real[j * rows + i] = cuCabs(data[j * rows + i]) / rows;
  }
}

void hilbert2(const double *x, double *out, const int rows, const int cols) {
  cufftHandle plan_f;
  cufftHandle plan_b;
  const int rank = 1;
  int n[] = {rows};
  const int dist = rows;
  const int stride = 1;

  cufftDoubleComplex *d_cx;
  double *d_real;
  HANDLE_CUDA_ERROR(
      cudaMalloc(&d_cx, rows * cols * sizeof(cufftDoubleComplex)));
  HANDLE_CUDA_ERROR(cudaMalloc(&d_real, rows * cols * sizeof(double)));

  HANDLE_CUFFT_ERROR(cufftPlanMany(&plan_f, rank, n, NULL, stride, dist, NULL,
                                   stride, dist, CUFFT_D2Z, cols));

  HANDLE_CUFFT_ERROR(cufftPlanMany(&plan_b, rank, n, NULL, stride, dist, NULL,
                                   stride, dist, CUFFT_Z2D, cols));

  HANDLE_CUDA_ERROR(cudaMemcpy(d_real, x, rows * cols * sizeof(double),
                               cudaMemcpyHostToDevice));

  // Forward fft
  HANDLE_CUFFT_ERROR(cufftExecD2Z(plan_f, d_real, d_cx));

  // Manipulate spectrum for Hilbert transform
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernelHilbert2<<<numBlocks, threadsPerBlock>>>(d_cx, rows, cols);

  // Backward fft
  HANDLE_CUFFT_ERROR(cufftExecZ2D(plan_b, d_cx, d_real));

  const double fct = 1. / rows;
  kernelAbs2<<<numBlocks, threadsPerBlock>>>(d_cx, d_real, rows, cols);

  // Copy to output
  HANDLE_CUDA_ERROR(cudaMemcpy(out, d_real, rows * cols * sizeof(double),
                               cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  // Cleanup
  HANDLE_CUDA_ERROR(cudaFree(d_cx));
  HANDLE_CUDA_ERROR(cudaFree(d_real));
  HANDLE_CUFFT_ERROR(cufftDestroy(plan_f));
  HANDLE_CUFFT_ERROR(cufftDestroy(plan_b));
}
