#pragma once

#include <thrust/device_vector.h>
#include <uspam/recon.hpp>

namespace uspam::cuda {

void recon_device(const double *device_in, const double *device_kernel,
                  double *device_out, int batchSize, int size, int kernelSize,
                  cudaStream_t stream = nullptr);

void recon_device(const thrust::device_vector<double> &device_in,
                  const thrust::device_vector<double> &device_kernel,
                  thrust::device_vector<double> &device_out, int batchSize,
                  int size, int kernelSize, cudaStream_t stream = nullptr);

void reconOneScan_device(const recon::ReconParams &params,
                         arma::Mat<double> &rf, arma::Mat<double> &rfLog,
                         bool flip, cudaStream_t stream);

} // namespace uspam::cuda
