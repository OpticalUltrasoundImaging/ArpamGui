#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>

namespace uspam::cuda {

// FIR filter
void firFilt2_same_device(const double *in, const double *kernel, double *out,
                          int inSize, int batchSize, int kernelSize,
                          cudaStream_t stream = nullptr);

// Reference implementation of Hilbert transform + abs
void hilbert2_ref(const double *in, double *out, int fftSize, int batchSize,
                  cudaStream_t stream = nullptr);

// Optimized implementation of Hilbert transform + abs
void hilbert2(const double *in, double *out, int fftSize, int batchSize,
              cudaStream_t stream = nullptr);

// Optimized implementation of Hilbert transform + abs that takes device arrays
// as input and output. Supports in place (device_in == device_out)
void hilbert2_device(const double *device_in, double *device_out, int fftSize,
                     int batchSize, cudaStream_t stream = nullptr);

void logCompress_device(const double *d_in, double *d_out, int size,
                        double noiseFloor, double desiredDynamicRangeDB,
                        cudaStream_t stream = nullptr);
void logCompress_device(const thrust::device_vector<double> &d_in,
                        thrust::device_vector<double> &d_out, double noiseFloor,
                        double desiredDynamicRangeDB,
                        cudaStream_t stream = nullptr);

double calcDynamicRange_device(const double *d_in, int size, double noiseFloor,
                               cudaStream_t stream = nullptr);

} // namespace uspam::cuda
