/**
Everything here assumes column major for 2D arrays
*/
// NOLINTBEGIN(*-pointer-arithmetic, *-trailing-return-type, *-const-cast)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuComplex.h>
#include <cuda/std/cmath>
#include <cufft.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "uspam/cudaSignal.h"
#include "uspam/cudaUtil.h"
#include <array>
#include <map>
#include <tuple>

__global__ void convolve1DSame(const double *in, const double *kernel,
                               double *output, int inSize, int batchSize,
                               int kernelSize) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k_half = kernelSize / 2;
  double sum = 0;

  if (i < inSize && j < batchSize) {
    int offset = j * inSize;
    for (int k = 0; k < kernelSize; k++) {
      int idx = i + k_half - k; // Center the kernel on the current element
      if (idx >= 0 && idx < inSize) {
        sum += in[offset + idx] * kernel[k];
      }
    }
    output[offset + i] = sum;
  }
}

// FIR filter
void uspam::cuda::firFilt2_same_device(const double *in, const double *kernel,
                                       double *out, const int inSize,
                                       const int batchSize,
                                       const int kernelSize,
                                       cudaStream_t stream) {
  dim3 threadsPerBlock(32, 16);
  dim3 numBlocks((inSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (batchSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
  convolve1DSame<<<numBlocks, threadsPerBlock, 0, stream>>>(
      in, kernel, out, inSize, batchSize, kernelSize);
}

__global__ void kernelHilbert_r2c_freq_switch(cufftDoubleComplex *data,
                                              const uint32_t rows,
                                              const uint32_t cols) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; // row
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; // col
  if (i < rows && i < cols) {
    const uint32_t idx = j * rows + i;
    // data[idx] *= -1j;
    data[idx] = cuCmul(data[idx], make_cuDoubleComplex(0, -1));
  }
}

__global__ void kernelHilbert_r2c_scale_and_abs(const double *real,
                                                const double *imag, double *out,
                                                const uint32_t rows,
                                                const uint32_t cols) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) {
    const uint32_t idx = j * rows + i;
    out[idx] = cuCabs(
        make_cuDoubleComplex(real[idx], imag[idx] / static_cast<double>(rows)));
  }
}

void uspam::cuda::hilbert2_ref(const double *in, double *out, const int fftSize,
                               const int batchSize, cudaStream_t stream) {
  cufftHandle planr2c{};
  cufftHandle planc2r{};

  std::array<int, 1> _fftSize = {fftSize};
  const int dist = fftSize;
  const int stride = 1;

  CUFFT_CALL(cufftCreate(&planr2c));
  CUFFT_CALL(cufftCreate(&planc2r));
  // clang-format off
  CUFFT_CALL(cufftPlanMany(&planr2c, _fftSize.size(), _fftSize.data(),
                           nullptr, stride, dist,
                           nullptr, stride, dist,
                           CUFFT_D2Z, batchSize));

  CUFFT_CALL(cufftPlanMany(&planc2r, _fftSize.size(), _fftSize.data(),
                           nullptr, stride, dist,
                           nullptr, stride, dist,
                           CUFFT_Z2D, batchSize));
  // clang-format on
  CUFFT_CALL(cufftSetStream(planr2c, stream));
  CUFFT_CALL(cufftSetStream(planc2r, stream));

  // Create device arrays
  cufftDoubleComplex *d_cx = nullptr;
  double *d_real = nullptr;
  double *d_imag = nullptr;

  CUDA_RT_CALL(
      cudaMalloc(&d_cx, fftSize * batchSize * sizeof(cufftDoubleComplex)));
  CUDA_RT_CALL(cudaMalloc(&d_real, fftSize * batchSize * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc(&d_imag, fftSize * batchSize * sizeof(double)));

  CUDA_RT_CALL(cudaMemcpyAsync(d_real, in, fftSize * batchSize * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

  // Forward fft
  CUFFT_CALL(cufftExecD2Z(planr2c, d_real, d_cx));

  // Manipulate spectrum for Hilbert transform
  dim3 threadsPerBlock(32, 16);
  dim3 numBlocks((fftSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (batchSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernelHilbert_r2c_freq_switch<<<numBlocks, threadsPerBlock, 0, stream>>>(
      d_cx, fftSize, batchSize);

  // Backward fft
  CUFFT_CALL(cufftExecZ2D(planc2r, d_cx, d_imag));

  kernelHilbert_r2c_scale_and_abs<<<numBlocks, threadsPerBlock, 0, stream>>>(
      d_real, d_imag, d_real, fftSize, batchSize);

  // Copy to output
  CUDA_RT_CALL(cudaMemcpyAsync(out, d_real,
                               fftSize * batchSize * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));

  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  // Cleanup
  CUDA_RT_CALL(cudaFree(d_cx));
  CUDA_RT_CALL(cudaFree(d_real));
  CUDA_RT_CALL(cudaFree(d_imag));

  CUDA_RT_CALL(cufftDestroy(planr2c));
  CUDA_RT_CALL(cufftDestroy(planc2r));
}

// fft_size, batch_size
using HilbertPlanKey = std::tuple<int, int>;
struct HilbertPlan {
  // Plans
  cufftHandle planr2c{};
  cufftHandle planc2r{};

  // Device arrays
  cufftDoubleComplex *d_cx{};
  double *d_real{};
  double *d_imag{};

  explicit HilbertPlan(const HilbertPlanKey &key) {
    const auto [_fftSize, batch_size] = key;

    std::array<int, 1> fftSize = {_fftSize};
    const int dist = _fftSize;
    const int stride = 1;

    CUFFT_CALL(cufftCreate(&planr2c));
    CUFFT_CALL(cufftCreate(&planc2r));

    // clang-format off
    CUFFT_CALL(cufftPlanMany(&planr2c, fftSize.size(), fftSize.data(),
                             nullptr, stride, dist,
                             nullptr, stride, dist,
                             CUFFT_D2Z, batch_size));
    CUFFT_CALL(cufftPlanMany(&planc2r, fftSize.size(), fftSize.data(),
                             nullptr, stride, dist,
                             nullptr, stride, dist,
                             CUFFT_Z2D, batch_size));
    // clang-format on

    const auto N = _fftSize * batch_size;
    CUDA_RT_CALL(cudaMalloc(&d_cx, N * sizeof(cufftDoubleComplex)));
    CUDA_RT_CALL(cudaMalloc(&d_real, N * sizeof(double)));
    CUDA_RT_CALL(cudaMalloc(&d_imag, N * sizeof(double)));
  }
  HilbertPlan(HilbertPlan &) = delete;
  HilbertPlan(HilbertPlan &&) = delete;
  HilbertPlan &operator=(HilbertPlan &) = delete;
  HilbertPlan &operator=(HilbertPlan &&) = delete;

  ~HilbertPlan() {
    CUDA_RT_CALL(cudaFree(d_cx));
    CUDA_RT_CALL(cudaFree(d_real));
    CUDA_RT_CALL(cudaFree(d_imag));

    CUDA_RT_CALL(cufftDestroy(planr2c));
    CUDA_RT_CALL(cufftDestroy(planc2r));
  }

  void setStream(cudaStream_t stream) const {
    CUFFT_CALL(cufftSetStream(planr2c, stream));
    CUFFT_CALL(cufftSetStream(planc2r, stream));
  }
};

template <class Key, class Val> auto get_cached(const Key &key) {
  static thread_local std::map<Key, std::unique_ptr<Val>> cache;
  auto &val = cache[key];
  if (val == nullptr) {
    val = std::make_unique<Val>(key);
  }
  return val.get();
}

void uspam::cuda::hilbert2(const double *in, double *out, const int fftSize,
                           const int batchSize, cudaStream_t stream) {
  auto &plan = *get_cached<HilbertPlanKey, HilbertPlan>({fftSize, batchSize});
  plan.setStream(stream);

  CUDA_RT_CALL(cudaMemcpyAsync(plan.d_real, in,
                               fftSize * batchSize * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

  // Forward fft
  CUFFT_CALL(cufftExecD2Z(plan.planr2c, plan.d_real, plan.d_cx));

  // Manipulate spectrum for Hilbert transform
  dim3 threadsPerBlock(32, 16);
  dim3 numBlocks((fftSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (batchSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernelHilbert_r2c_freq_switch<<<numBlocks, threadsPerBlock, 0, stream>>>(
      plan.d_cx, fftSize, batchSize);

  // Backward fft
  CUFFT_CALL(cufftExecZ2D(plan.planc2r, plan.d_cx, plan.d_imag));

  kernelHilbert_r2c_scale_and_abs<<<numBlocks, threadsPerBlock, 0, stream>>>(
      plan.d_real, plan.d_imag, plan.d_real, fftSize, batchSize);

  // Copy to output
  CUDA_RT_CALL(cudaMemcpyAsync(out, plan.d_real,
                               fftSize * batchSize * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
}

void uspam::cuda::hilbert2_device(const double *device_in, double *device_out,
                                  const int fftSize, const int batchSize,
                                  cudaStream_t stream) {
  auto &plan = *get_cached<HilbertPlanKey, HilbertPlan>({fftSize, batchSize});
  plan.setStream(stream);

  // Forward fft
  CUFFT_CALL(
      cufftExecD2Z(plan.planr2c, const_cast<double *>(device_in), plan.d_cx));

  // Manipulate spectrum for Hilbert transform
  dim3 threadsPerBlock(32, 16);
  dim3 numBlocks((fftSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (batchSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernelHilbert_r2c_freq_switch<<<numBlocks, threadsPerBlock, 0, stream>>>(
      plan.d_cx, fftSize, batchSize);

  // Backward fft
  CUFFT_CALL(cufftExecZ2D(plan.planc2r, plan.d_cx, plan.d_imag));

  kernelHilbert_r2c_scale_and_abs<<<numBlocks, threadsPerBlock, 0, stream>>>(
      device_in, plan.d_imag, device_out, fftSize, batchSize);
}

double calcDynamicRange_device(const double *d_in, int size, double noiseFloor,
                               cudaStream_t stream) {
  thrust::device_ptr<const double> in(d_in);

  const auto peakIter =
      thrust::max_element(thrust::cuda::par.on(stream), in, in + size);
  const double peakLevel = *peakIter;
  const double dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);

  return dynamicRangeDB;
}

void uspam::cuda::logCompress_device(const double *d_in, double *d_out,
                                     int size, double noiseFloor,
                                     double desiredDynamicRangeDB,
                                     cudaStream_t stream) {
  thrust::device_ptr<const double> in(d_in);
  thrust::device_ptr<double> out(d_out);

  // Apply log compression with clipping
  thrust::transform(thrust::cuda::par.on(stream), in, in + size, out,
                    [=] __device__(const double val) {
                      double normVal = val / noiseFloor;
                      double compVal = (normVal > 0 ? 20 * log10(normVal) : 0);
                      compVal = max(compVal, 0.0);
                      compVal = min(compVal, desiredDynamicRangeDB);
                      return compVal / desiredDynamicRangeDB;
                    });
}

void uspam::cuda::logCompress_device(const thrust::device_vector<double> &d_in,
                                     thrust::device_vector<double> &d_out,
                                     double noiseFloor,
                                     double desiredDynamicRangeDB,
                                     cudaStream_t stream) {

  d_out.resize(d_in.size());
  logCompress_device(thrust::raw_pointer_cast(d_in.data()),
                     thrust::raw_pointer_cast(d_out.data()), d_in.size(),
                     noiseFloor, desiredDynamicRangeDB, stream);
}

// __global__ void kernelLogCompress(double *d_in, double *d_out, int size,
//                                   double noiseFloor,
//                                   double desiredDynamicRangeDB) {
//   const int i = blockIdx.x * blockDim.x + threadIdx.x;

//   if (i < size) {
//     const double val = d_in[i];
//     double compressed = 20.0 * log10(val / noiseFloor);
//     compressed = max(compressed, double(0));
//     compressed = min(compressed, desiredDynamicRangeDB);
//     d_out[i] = compressed / desiredDynamicRangeDB;
//   }
// }

// double uspam::cuda::logCompress_device(double *_in, double *_out, int size,
//                                        double noiseFloor,
//                                        double desiredDynamicRangeDB,
//                                        cudaStream_t stream) {
//   // cudaStream_t streamMax;
//   // CUDA_RT_CALL(cudaStreamCreate(&streamMax));

//   // const double peakLevel =
//   //     *thrust::max_element(thrust::cuda::par.on(streamMax), _in, _in +
//  size);
//  // const double dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);
//  const double dynamicRangeDB = 0.;

//  int blockSize = 512;
//  int numBlocks = (size + blockSize - 1) / blockSize;
//  kernelLogCompress<<<numBlocks, blockSize, 0, stream>>>(
//      _in, _out, size, noiseFloor, desiredDynamicRangeDB);

//  // CUDA_RT_CALL(cudaGetLastError());
//  // CUDA_RT_CALL(cudaStreamDestroy(streamMax));

//  return dynamicRangeDB;
// }

// NOLINTEND(*-pointer-arithmetic, *-trailing-return-type, *-const-cast)