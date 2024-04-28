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

void reconOneScan_device(const recon::ReconParams2 &params,
                         io::PAUSpair<double> &rf, io::PAUSpair<double> &rfLog,
                         bool flip, cudaStream_t stream) {

  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf.PA);
    imutil::fliplr_inplace(rf.US);

    // Do rotate
    const auto rotate_offset = params.aline_rotation_offset;
    rf.PA = arma::shift(rf.PA, rotate_offset, 1);
    rf.US = arma::shift(rf.US, rotate_offset, 1);
  }

  // compute filter kernels and move to device
  const auto kernelPA =
      signal::firwin2(95, params.filter_freq_PA, params.filter_gain_PA);
  const auto kernelUS =
      signal::firwin2(95, params.filter_freq_US, params.filter_gain_US);

  thrust::device_vector<double> kernelPA_device(kernelPA.size());
  thrust::device_vector<double> kernelUS_device(kernelUS.size());
  copy_async_host2device(kernelPA_device, kernelPA, stream);
  copy_async_host2device(kernelUS_device, kernelUS, stream);

  // Move RF to device
  thrust::device_vector<double> rf_PA;
  thrust::device_vector<double> rf_US;
  copy_async_host2device(rf_PA, rf.PA, stream);
  copy_async_host2device(rf_US, rf.US, stream);

  // allocate result device buffer
  thrust::device_vector<double> rf_env_PA(rf_PA.size());
  thrust::device_vector<double> rf_env_US(rf_US.size());
  thrust::device_vector<double> rf_log_PA(rf_PA.size());
  thrust::device_vector<double> rf_log_US(rf_US.size());

  // Wait for copies to finish
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  // recon
  recon_device(rf_PA, kernelPA_device, rf_env_PA, rf.PA.n_cols, rf.PA.n_rows,
               kernelPA_device.size(), stream);
  recon_device(rf_US, kernelUS_device, rf_env_US, rf.US.n_cols, rf.US.n_rows,
               kernelUS_device.size(), stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  logCompress_device(rf_env_PA, rf_log_PA, params.noise_floor_PA,
                     params.desired_dynamic_range_PA);
  logCompress_device(rf_env_US, rf_log_US, params.noise_floor_US,
                     params.desired_dynamic_range_US);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  // Copy result to host
  copy_async_device2host(rfLog.PA, rf_log_PA, stream);
  copy_async_device2host(rfLog.US, rf_log_US, stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
}

} // namespace uspam::cuda
