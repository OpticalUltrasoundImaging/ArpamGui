#pragma once

#include <Eigen/Dense>

#include "fftconv.hpp"
#include "libarpam/fft.hpp"
#include "libarpam/signal.hpp"

namespace arpam::recon {

namespace pipeline {

class FIRFilter {
public:
  FIRFilter(size_t n, int numtaps, const Eigen::ArrayXd &freq,
            const Eigen::ArrayXd &gain)
      : kernel(signal::firwin2(numtaps, freq, gain)) {}

  template <typename T>
  void forward(const std::span<const T> input, const std::span<T> output) {
    fftconv::oaconvolve_fftw_same(
        std::span<const double>(input.data(), input.size()),
        std::span<const double>(kernel.data(), kernel.size()),
        std::span<double>(output.data(), output.size()));
  }

  template <typename T>
  void forward(const Eigen::MatrixX<T> &input, Eigen::MatrixX<T> &output) {
    for (auto col_i = 0; col_i < input.cols(); col_i++) {
      //   this->forward(input.col(col_i), output.col(col_i));
      const auto _inp = input.col(col_i);
      auto _out = output.col(col_i);
      this->forward(std::span<const double>(_inp.data(), _inp.size()),
                    std::span<double>(_out.data(), _out.size()));
    }
  }

private:
  Eigen::ArrayXd kernel;
};

class EnvelopeDetection {
public:
  explicit EnvelopeDetection(Eigen::Index n) : engine(n) {}

  void forward(const std::span<const double> input, std::span<double> output) {
    assert(input.size() == output.size());
    const int64_t n = input.size();

    // Copy input to real buffer
    // NOLINTBEGIN(*-pointer-arithmetic)
    for (int i = 0; i < n; ++i) {
      engine.in[i][0] = input[i];
      engine.in[i][1] = 0.;
    }
    // NOLINTEND(*-pointer-arithmetic)

    // Execute r2c fft
    engine.execute_forward();

    // NOLINTBEGIN(*-pointer-arithmetic)
    // Zero negative frequencies (half-Hermitian to Hermitian conversion)
    // Double the magnitude of positive frequencies
    const auto n_half = n / 2;
    for (auto i = 1; i < n_half; ++i) {
      engine.out[i][0] *= 2;
      engine.out[i][1] *= 2;
    }

    if (n % 2 == 0) {
      engine.out[n_half][0] = 0;
      engine.out[n_half][1] = 0;
    } else {
      engine.out[n_half][0] *= 2;
      engine.out[n_half][1] *= 2;
    }

    for (auto i = n_half + 1; i < n; ++i) {
      engine.out[i][0] = 0;
      engine.out[i][1] = 0;
    }
    // NOLINTEND(*-pointer-arithmetic)

    // Execute c2r fft on modified spectrum
    engine.execute_backward();

    // Construct the analytic signal
    Eigen::ArrayXcd analytic_signal(n);
    const double fct = 1. / static_cast<double>(n);
    for (Eigen::Index i = 0; i < n; ++i) {
      const auto real = input[i];
      const auto imag = engine.in[i][1] * fct;
      output[i] = std::abs(std::complex{real, imag});
    }
  };

private:
  fft ::fftw_engine_1d engine;
};

} // namespace pipeline

/*
Filt
*/
void recon(const Eigen::MatrixXd &rf, Eigen::MatrixXd &output);

} // namespace arpam::recon