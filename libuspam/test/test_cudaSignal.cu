#include <armadillo>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fftconv.hpp>
#include <fstream>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>

#include "uspam/cudaSignal.cuh"
#include "uspam/cudaUtil.cuh"
#include "uspam/recon.hpp"
#include "uspam/signal.hpp"
#include "uspam/timeit.hpp"

namespace signal = uspam::signal;

TEST(CudaHilbertTest, Correct) {
  // const arma::mat input(1024, 1024, arma::fill::randn);
  const int N = 1024;
  const arma::mat input = [&]() {
    arma::mat input(N, N, arma::fill::none);
    const arma::vec t = arma::linspace(0, 2 * arma::datum::pi, N);
    // Generate each column as a sine wave
    // with a carrier wave of increasing frequency
    for (int j = 0; j < N; ++j) {
      const double freq = 1 + j;
      input.col(j) = arma::sin(freq * t) % arma::sin(t);
    }
    return input;
  }();

  const arma::mat expected = [&]() {
    arma::mat expected(N, N, arma::fill::none);
    for (int i = 0; i < expected.n_cols; ++i) {
      const auto src = input.unsafe_col(i);
      auto dst = expected.unsafe_col(i);
      signal::hilbert_abs(src, dst);
    }
    return expected;
  }();

  arma::mat output(N, N, arma::fill::zeros);
  uspam::cuda::hilbert2(input.memptr(), output.memptr(), input.n_rows,
                        input.n_cols);

  // input.save("hilbert_input.bin", arma::raw_binary);
  // expected.save("hilbert_expected.bin", arma::raw_binary);
  // output.save("hilbert_output.bin", arma::raw_binary);

  for (int j = 0; j < expected.n_cols; ++j) {
    for (int i = 0; i < expected.n_rows; ++i) {
      EXPECT_NEAR(output(i, j), expected(i, j), 1.5e-8);
    }
  }
}

TEST(CudaHilbertTest, Bench) {
  const int fft_size = 8192;
  const int batch_size = 1000;
  arma::mat input(fft_size, batch_size, arma::fill::randn);

  const int n_runs = 20;
  {
    arma::mat env(fft_size, batch_size, arma::fill::none);
    auto nanos = bench(
        "hilbert_abs_r2c CPU", n_runs,
        [&]() {
          for (int i = 0; i < input.n_cols; ++i) {
            const auto src = input.unsafe_col(i);
            auto dst = env.unsafe_col(i);
            signal::hilbert_abs_r2c(src, dst);
          }
        },
        true);
  }

  {
    cudaStream_t stream;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    arma::mat env(fft_size, batch_size, arma::fill::none);
    bench(
        "hilbert_abs_r2c CUDA", n_runs,
        [&]() {
          uspam::cuda::hilbert2(input.memptr(), env.memptr(), input.n_rows,
                                input.n_cols, stream);
          CUDA_RT_CALL(cudaStreamSynchronize(stream));
        },
        true);

    CUDA_RT_CALL(cudaStreamDestroy(stream));
  }

  {
    cudaStream_t stream;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    thrust::device_vector<double> device_in(fft_size * batch_size);
    thrust::device_vector<double> device_out(fft_size * batch_size);
    bench(
        "hilbert_abs_r2c CUDA device", n_runs,
        [&]() {
          uspam::cuda::hilbert2_device(
              thrust::raw_pointer_cast(device_in.data()),
              thrust::raw_pointer_cast(device_out.data()), fft_size, batch_size,
              stream);
          CUDA_RT_CALL(cudaStreamSynchronize(stream));
        },
        true);

    CUDA_RT_CALL(cudaStreamDestroy(stream));
  }
}

TEST(CudaFIRFilterTest, Correct) {
  const int N = 1024;
  const int batchSize = 1024;
  const int Nkernel = 65;

  const arma::mat in = [&]() {
    arma::mat input(N, batchSize, arma::fill::none);
    const arma::vec t = arma::linspace(0, 2 * arma::datum::pi, N);
    // Generate each column as a sine wave
    // with a carrier wave of increasing frequency
    for (int j = 0; j < batchSize; ++j) {
      const double freq = 1 + j;
      input.col(j) = arma::sin(freq * t) % arma::sin(t);
    }
    return input;
  }();

  const arma::vec kernel = [&]() {
    const arma::vec t = arma::linspace(0, 2 * arma::datum::pi, Nkernel);
    arma::vec kernel = arma::sin(t);
    return kernel;
  }();

  const arma::mat expected = [&]() {
    arma::mat expected(N, batchSize, arma::fill::none);
    for (int i = 0; i < batchSize; ++i) {
      expected.col(i) = arma::conv(in.col(i), kernel, "same");
    }
    return expected;
  }();

  thrust::device_vector<double> device_in(in.begin(), in.end());
  thrust::device_vector<double> device_kernel(kernel.begin(), kernel.end());
  thrust::device_vector<double> device_out(N * batchSize);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  uspam::cuda::firFilt2_same_device(
      thrust::raw_pointer_cast(device_in.data()),
      thrust::raw_pointer_cast(device_kernel.data()),
      thrust::raw_pointer_cast(device_out.data()), N, batchSize,
      device_kernel.size(), stream);
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  thrust::host_vector<double> out = device_out;
  const arma::mat _out(thrust::raw_pointer_cast(out.data()), N, batchSize,
                       false, true);

  in.save("firfilt_in.bin", arma::raw_binary);
  kernel.save("firfilt_kernel.bin", arma::raw_binary);
  expected.save("firfilt_expected.bin", arma::raw_binary);
  _out.save("firfilt_out.bin", arma::raw_binary);

  for (int j = 0; j < batchSize; ++j) {
    for (int i = 0; i < N; ++i) {
      EXPECT_NEAR(_out(i, j), expected(i, j), 1.5e-8);
    }
  }

  CUDA_RT_CALL(cudaStreamDestroy(stream));
}

TEST(CudaFIRFilterTest, Bench) {
  const int N = 8192;
  const int batchSize = 1000;
  const int Nkernel = 95;

  const arma::mat in = [&]() {
    arma::mat input(N, batchSize, arma::fill::none);
    const arma::vec t = arma::linspace(0, 2 * arma::datum::pi, N);
    // Generate each column as a sine wave
    // with a carrier wave of increasing frequency
    for (int j = 0; j < batchSize; ++j) {
      const double freq = 1 + j;
      input.col(j) = arma::sin(freq * t) % arma::sin(t);
    }
    return input;
  }();

  const arma::vec kernel = [&]() {
    const arma::vec t = arma::linspace(0, 2 * arma::datum::pi, Nkernel);
    arma::vec kernel = arma::sin(t);
    return kernel;
  }();

  int runs = 10;
  arma::mat out(N, batchSize, arma::fill::none);
  // bench(
  //     "firfilt ARMA", runs,
  //     [&]() {
  //       for (int i = 0; i < batchSize; ++i) {
  //         out.col(i) = arma::conv(in.col(i), kernel, "same");
  //       }
  //     },
  //     true);

  runs = 100;
  bench(
      "firfilt fftconv CPU", runs,
      [&]() {
        for (int i = 0; i < batchSize; ++i) {
          const auto src = in.unsafe_col(i);
          auto dst = out.unsafe_col(i);
          fftconv::oaconvolve_fftw_same<double>(src, kernel, dst);
        }
      },
      true);

  thrust::device_vector<double> device_in(in.begin(), in.end());
  thrust::device_vector<double> device_kernel(kernel.begin(), kernel.end());
  thrust::device_vector<double> device_out(N * batchSize);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  bench(
      "firfilt CUDA", runs,
      [&]() {
        uspam::cuda::firFilt2_same_device(
            thrust::raw_pointer_cast(device_in.data()),
            thrust::raw_pointer_cast(device_kernel.data()),
            thrust::raw_pointer_cast(device_out.data()), N, batchSize,
            device_kernel.size(), stream);
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
      },
      true);

  CUDA_RT_CALL(cudaStreamDestroy(stream));
}

TEST(CudaLogCompressTest, Correct) {
  const int N = 8192;
  arma::vec in(N, arma::fill::randu);
  in = 20 * in + 1;

  arma::vec expected(N, arma::fill::none);
  uspam::recon::logCompress<double>(in, expected, 1., 5.);

  thrust::device_vector<double> d_in(in.begin(), in.end());
  thrust::device_vector<double> d_out(N);
  uspam::cuda::logCompress_device(thrust::raw_pointer_cast(d_in.data()),
                                  thrust::raw_pointer_cast(d_out.data()), N, 1,
                                  5);

  arma::vec out(N, arma::fill::none);
  thrust::copy(d_out.begin(), d_out.end(), out.begin());

  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR(out(i), expected(i), 1.5e-8);
  }
}

TEST(CudaLogCompressTest, Bench) {
  const int N = 8192 * 1000;
  arma::vec in(N, arma::fill::randu);
  in = 20 * in + 1;

  arma::vec expected(N, arma::fill::none);

  int runs = 100;

  bench(
      "logCompress CPU", runs,
      [&]() { uspam::recon::logCompress<double>(in, expected, 1., 5.); }, true);

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  thrust::device_vector<double> d_in(in.begin(), in.end());
  // thrust::device_vector<double> d_in(N);
  // thrust::copy(thrust::cuda::par.on(stream), in.memptr(), in.memptr() + N,
  // d_in.begin());
  thrust::device_vector<double> d_out(N);

  bench(
      "logCompress CUDA", runs,
      [&]() {
        uspam::cuda::logCompress_device(thrust::raw_pointer_cast(d_in.data()),
                                        thrust::raw_pointer_cast(d_out.data()),
                                        N, 1, 5);
      },
      true);

  CUDA_RT_CALL(cudaStreamDestroy(stream));
}

TEST(CudaReconTest, Correct) {
  // TODO
}

TEST(CudaReconTest, Bench) {
  const int size = 8192;
  const int batchSize = 1000;
  const int kernelSize = 95;
  arma::mat input(size, batchSize, arma::fill::randn);
  arma::vec kernel(kernelSize, arma::fill::randu);

  const int n_runs = 20;
  {
    arma::mat env(size, batchSize, arma::fill::none);
    auto nanos = bench(
        "recon CPU", n_runs, [&]() { uspam::recon::recon(input, kernel, env); },
        true);
  }

  {
    cudaStream_t stream;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    thrust::device_vector<double> device_in(size * batchSize);
    thrust::device_vector<double> device_out(size * batchSize);
    thrust::device_vector<double> device_kernel(kernelSize);

    bench(
        "recon CUDA", n_runs,
        [&]() {
          uspam::cuda::recon_device(
              thrust::raw_pointer_cast(device_in.data()),
              thrust::raw_pointer_cast(device_kernel.data()),
              thrust::raw_pointer_cast(device_out.data()), size, kernelSize,
              batchSize, stream);
          CUDA_RT_CALL(cudaStreamSynchronize(stream));
        },
        true);

    CUDA_RT_CALL(cudaStreamDestroy(stream));
  }
}
