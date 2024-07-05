#include "uspam/cudaRecon.h"
#include "uspam/cudaSignal.h"
#include "uspam/cudaUtil.h"
#include "uspam/imutil.hpp"
#include "uspam/recon.hpp"
#include <thrust/device_vector.h>

namespace uspam::cuda {

void recon_device(const double *device_in, const double *kernel,
                  double *device_out, const int batchSize, const int size,
                  const int kernelSize, cudaStream_t stream) {

  firFilt2_same_device(device_in, kernel, device_out, size, batchSize,
                       kernelSize, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  hilbert2_device(device_out, device_out, size, batchSize, stream);
}

void recon_device(const thrust::device_vector<double> &device_in,
                  const thrust::device_vector<double> &device_kernel,
                  thrust::device_vector<double> &device_out, int batchSize,
                  int size, int kernelSize, cudaStream_t stream) {
  recon_device(thrust::raw_pointer_cast(device_in.data()),
               thrust::raw_pointer_cast(device_kernel.data()),
               thrust::raw_pointer_cast(device_out.data()), batchSize, size,
               kernelSize, stream);
}

void reconOneScan_device(const recon::ReconParams &params,
                         arma::Mat<double> &rf, arma::Mat<double> &rfLog,
                         bool flip, cudaStream_t stream) {

  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf);
    imutil::fliplr_inplace(rf);

    // Do rotate
    const auto rotate_offset = params.rotateOffset;
    rf = arma::shift(rf, rotate_offset, 1);
  }

  // compute filter kernels and move to device
  const auto kernel = signal::firwin2(95, params.filterFreq, params.filterGain);

  thrust::device_vector<double> kernel_device(kernel.size());
  copy_async_host2device(kernel_device, kernel, stream);

  // Move RF to device
  thrust::device_vector<double> rf_device;
  thrust::device_vector<double> rf_US;
  copy_async_host2device(rf_device, rf, stream);

  // allocate result device buffer
  thrust::device_vector<double> rf_env(rf_device.size());
  thrust::device_vector<double> rf_log(rf_device.size());

  // Wait for copies to finish
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  // recon
  recon_device(rf_device, kernel_device, rf_env, rf.n_cols, rf.n_rows,
               kernel_device.size(), stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  logCompress_device(rf_env, rf_log, params.noiseFloor_mV,
                     params.desiredDynamicRange);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  // Copy result to host
  copy_async_device2host(rfLog, rf_log, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
}

void reconOneScan_device(const recon::ReconParams2 &params,
                         io::PAUSpair<double> &rf, io::PAUSpair<double> &rfLog,
                         bool flip, cudaStream_t stream) {

  reconOneScan_device(params.PA, rf.PA, rfLog.PA, flip, stream);
  reconOneScan_device(params.US, rf.US, rfLog.US, flip, stream);
}

} // namespace uspam::cuda
