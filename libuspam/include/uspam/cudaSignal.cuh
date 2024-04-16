#pragma once

#include "uspam/cudaUtil.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

namespace uspam {
namespace cuda {

// FIR filter
void firFilt2_same_device(const double *in, const double *kernel, double *out,
                          const int inSize, const int batchSize,
                          const int kernelSize,
                          cudaStream_t stream = (cudaStream_t)0);

// Reference implementation of Hilbert transform + abs
void hilbert2_ref(const double *in, double *out, const int fftSize,
                  const int batchSize, cudaStream_t stream = (cudaStream_t)0);

// Optimized implementation of Hilbert transform + abs
void hilbert2(const double *in, double *out, const int fftSize,
              const int batchSize, cudaStream_t stream = (cudaStream_t)0);

// Optimized implementation of Hilbert transform + abs that takes device arrays
// as input and output. Supports in place (device_in == device_out)
void hilbert2_device(const double *device_in, double *device_out,
                     const int fftSize, const int batchSize,
                     cudaStream_t stream = (cudaStream_t)0);

double logCompress_device(double *d_in, double *d_out, int size,
                          double noiseFloor, double desiredDynamicRangeDB,
                          cudaStream_t stream = (cudaStream_t)0);

void recon_device(const double *device_in, const double *device_kernel,
                  double *device_out, const int size, const int kernelSize,
                  const int batchSize, cudaStream_t stream = (cudaStream_t)0);

} // namespace cuda
} // namespace uspam
