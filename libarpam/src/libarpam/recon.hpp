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
  double dynamic_range{25.0};
};

namespace pipeline {

class FIRFilter {
public:
  using T = double;

  FIRFilter(size_t n, const FIRFilterParams<T> &params) { setKernel(params); }

  void forward(std::span<const T> input, std::span<T> output) const;
  void forward(const Eigen::MatrixX<T> &input, Eigen::MatrixX<T> &output) const;

  inline void setKernel(const Eigen::ArrayX<T> &kernel) { m_kernel = kernel; }
  void setKernel(const FIRFilterParams<T> &params);

private:
  Eigen::ArrayX<T> m_kernel;
};

class EnvelopeDetection {
public:
  explicit EnvelopeDetection(size_t n) : engine(n) {}

  void forward(std::span<const double> input, std::span<double> output);

private:
  fft::fftw_engine_1d engine;
};

// TODO(tnie): unit test
class LogCompression {
public:
  explicit LogCompression(size_t n, double db) : m_db(db) {}
  inline void set_db(double db) { m_db = db; };

  void forward(std::span<const double> input, std::span<double> output) const;
  void forward(std::span<const double> input, std::span<uint8_t> output) const;

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