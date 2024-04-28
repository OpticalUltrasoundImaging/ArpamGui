#pragma once

#include <cstdio>

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    auto status = static_cast<cudaError_t>(call);                              \
    if (status != cudaSuccess)                                                 \
      fprintf(stderr,                                                          \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "       \
              "with "                                                          \
              "%s (%d).\n",                                                    \
              #call, __LINE__, __FILE__, cudaGetErrorString(status), status);  \
  }
#endif // CUDA_RT_CALL

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                       \
  {                                                                            \
    auto status = static_cast<cufftResult>(call);                              \
    if (status != CUFFT_SUCCESS)                                               \
      fprintf(stderr,                                                          \
              "ERROR: CUFFT call \"%s\" in line %d of file %s failed "         \
              "with "                                                          \
              "code (%d).\n",                                                  \
              #call, __LINE__, __FILE__, status);                              \
  }
#endif // CUFFT_CALL

#include <armadillo>
#include <thrust/device_vector.h>

/***
 * Copy from host to device async using a cuda stream
 */
template <typename T>
void copy_async_host2device(thrust::device_vector<T> &dst,
                            const arma::Col<T> &src, cudaStream_t stream) {
  dst.resize(src.size());
  CUDA_RT_CALL(cudaMemcpyAsync(thrust::raw_pointer_cast(dst.data()),
                               src.memptr(), src.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
}
template <typename T>
void copy_async_host2device(thrust::device_vector<T> &dst,
                            const arma::Mat<T> &src, cudaStream_t stream) {
  dst.resize(src.size());
  CUDA_RT_CALL(cudaMemcpyAsync(thrust::raw_pointer_cast(dst.data()),
                               src.memptr(), src.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
}

/***
 * Copy from device to host async using a cuda stream
 */
template <typename T>
void copy_async_device2host(arma::Col<T> &dst,
                            const thrust::device_vector<T> &src,
                            cudaStream_t stream) {
  dst.resize(src.size());
  CUDA_RT_CALL(
      cudaMemcpyAsync(dst.memptr(), thrust::raw_pointer_cast(src.data()),
                      src.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
}
template <typename T>
void copy_async_device2host(arma::Mat<T> &dst,
                            const thrust::device_vector<T> &src,
                            cudaStream_t stream) {
  assert(dst.size() == src.size());
  CUDA_RT_CALL(
      cudaMemcpyAsync(dst.memptr(), thrust::raw_pointer_cast(src.data()),
                      src.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
}