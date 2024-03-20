#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "fftconv.hpp"
#include "libarpam/fft.hpp"
#include "libarpam/signal.hpp"

namespace arpam::recon {

using fftconv::FloatOrDouble;

template <FloatOrDouble T> struct FIRFilterParams {
  int numtaps;
  Eigen::ArrayX<T> freq;
  Eigen::ArrayX<T> gain;

  [[nodiscard]] auto validate() const -> bool {
    if (numtaps < 3) {
      std::cerr << "numtaps must be positive and greater than 3\n";
      return false;
    }
    if (freq.size() != gain.size()) {
      std::cerr << "freq and gain must have the same size.\n";
      return false;
    }
    if (freq[0] != 0) {
      std::cerr << "freq[0] must be 0\n";
      return false;
    }

    return true;
  }
};

template <FloatOrDouble Real> struct ReconParams {
  FIRFilterParams<Real> firFilterParams;
  float dynamic_range{25.0F};
};

namespace pipeline {

template <FloatOrDouble T> class FIRFilter {
public:
  FIRFilter(size_t n, const FIRFilterParams<T> &params) { setKernel(params); }

  void forward(const std::span<const T> input, const std::span<T> output) {
    // Convolution with kernel using "same" padding
    assert(input.size() <= output.size());
    fftconv::oaconvolve_fftw_same<T>(
        input, std::span<const T>(m_kernel.data(), m_kernel.size()),
        std::span<T>(output.data(), input.size()));
  }

  void forward(const Eigen::MatrixX<T> &input, Eigen::MatrixX<T> &output) {
    for (auto col_i = 0; col_i < input.cols(); col_i++) {
      const auto in_col = input.col(col_i);
      auto out_col = output.col(col_i);
      this->forward(std::span<const T>(in_col.data(), in_col.size()),
                    std::span<T>(out_col.data(), out_col.size()));
    }
  }

  void setKernel(const Eigen::ArrayX<T> &kernel) { m_kernel = kernel; }

  void setKernel(const FIRFilterParams<T> &params) {
    if (!params.validate()) {
      throw std::runtime_error("Bad FIRFilterParams. Check stderr.");
    }

    if constexpr (std::is_same_v<T, double>) {
      m_kernel = signal::firwin2(params.numtaps, params.freq, params.gain);
    } else { // T == float
      m_kernel =
          signal::firwin2(params.numtaps, params.freq.template cast<double>(),
                          params.gain.template cast<double>())
              .template cast<float>();
    }
  }

private:
  Eigen::ArrayX<T> m_kernel;
};

class EnvelopeDetection {
public:
  explicit EnvelopeDetection(size_t n) : engine(n) {}

  void forward(const std::span<const double> input, std::span<double> output) {
    assert(input.size() == output.size());
    const auto n = static_cast<int64_t>(input.size());

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
  fft::fftw_engine_1d engine;
};

// TODO(tnie): unit test
class LogCompression {
public:
  explicit LogCompression(size_t n, double db) : m_db(db) {}
  inline void set_db(double db) { m_db = db; };
  inline void forward(const std::span<const double> input,
                      std::span<double> output) const {
    const auto max_val = *std::max_element(input.begin(), input.end());
    if (max_val != 0) {
      const auto scale = 1 / max_val;
      for (int i = 0; i < input.size(); ++i) {
        const double compressed =
            20.0 / m_db * (std::log10(input[i] * scale) + 1);
        output[i] = compressed < 0 ? 0 : compressed;
      }
    } else {
      std::fill(output.begin(), output.end(), 0);
    }
  }
  inline void forward(const std::span<const double> input,
                      std::span<uint8_t> output) const {
    const auto max_val = *std::max_element(input.begin(), input.end());
    if (max_val != 0) {
      const auto scale = 1 / max_val;
      for (int i = 0; i < input.size(); ++i) {
        const auto compressed = static_cast<uint8_t>(
            255.0 * (20.0 / m_db * (std::log10(input[i] * scale) + 1)));
        output[i] = compressed < 1 ? 0 : compressed;
      }
    } else {
      std::fill(output.begin(), output.end(), 0);
    }
  }

private:
  double m_db;
};

} // namespace pipeline

/*
Filt
*/
void recon(const ReconParams<double> &params, const Eigen::MatrixXd &rf,
           Eigen::MatrixXd &output);

void recon(const ReconParams<double> &params, const Eigen::MatrixXd &rf,
           Eigen::MatrixX<uint8_t> &output);
} // namespace arpam::recon