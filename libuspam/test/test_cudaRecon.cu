#include <cuda_runtime.h>

#include "uspam/cudaRecon.h"
#include "uspam/cudaUtil.h"
#include "uspam/recon.hpp"
#include "uspam/timeit.hpp"
#include <armadillo>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>

// NOLINTBEGIN(*-non-const-global-variables, modernize-*, *-magic-numbers)

TEST(CudaReconTest, Correct) {
  // test against CPU version
  const int size = 8192 / 2;
  const int batchSize = 1000;
  const int kernelSize = 95;
  const arma::mat input(size, batchSize, arma::fill::randn);
  const arma::vec kernel(kernelSize, arma::fill::randu);
  arma::mat env_expected(size, batchSize, arma::fill::none);

  uspam::recon::recon(input, kernel, env_expected);

  arma::mat env(size, batchSize, arma::fill::none);
  {
    cudaStream_t stream{};
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    thrust::device_vector<double> device_in(size * batchSize);
    thrust::device_vector<double> device_out(size * batchSize);
    thrust::device_vector<double> device_kernel(kernelSize);

    copy_async_host2device(device_in, input, stream);
    copy_async_host2device(device_kernel, kernel, stream);
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    uspam::cuda::recon_device(thrust::raw_pointer_cast(device_in.data()),
                              thrust::raw_pointer_cast(device_kernel.data()),
                              thrust::raw_pointer_cast(device_out.data()),
                              batchSize, size, kernelSize, stream);

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    copy_async_device2host(env, device_out, stream);
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
  }

  for (int j = 0; j < env.n_cols; ++j) {
    for (int i = 0; i < env.n_rows; ++i) {
      EXPECT_NEAR(env(i, j), env_expected(i, j), 1.5e-8);
    }
  }

  // ASSERT_TRUE(arma::approx_equal(env, env_expected, "absdiff", 1e-9));
}

TEST(CudaReconTest, Bench) {
  const int size = 8192;
  const int batchSize = 1000;
  const int kernelSize = 95;
  arma::mat input(size, batchSize, arma::fill::randn);
  arma::vec kernel(kernelSize, arma::fill::randu);

  int n_runs = 10;
  {
    arma::mat env_expected(size, batchSize, arma::fill::none);
    auto nanos = bench(
        "recon CPU", n_runs,
        [&]() { uspam::recon::recon(input, kernel, env_expected); }, true);
  }

  n_runs = 20;
  {
    cudaStream_t stream{};
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    thrust::device_vector<double> device_in(size * batchSize);
    thrust::device_vector<double> device_out(size * batchSize);
    thrust::device_vector<double> device_kernel(kernelSize);

    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    bench(
        "recon CUDA", n_runs,
        [&]() {
          uspam::cuda::recon_device(
              thrust::raw_pointer_cast(device_in.data()),
              thrust::raw_pointer_cast(device_kernel.data()),
              thrust::raw_pointer_cast(device_out.data()), batchSize, size,
              kernelSize, stream);
          CUDA_RT_CALL(cudaStreamSynchronize(stream));
        },
        true);

    CUDA_RT_CALL(cudaStreamDestroy(stream));
  }
}
// NOLINTEND(*-non-const-global-variables, modernize-*, *-magic-numbers)