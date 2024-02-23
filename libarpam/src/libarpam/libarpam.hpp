#pragma once

#include <iostream>

#include "fftconv.hpp"
#include <Eigen/Dense>
#include <fftw3.h>
#include <stdexcept>

#include "libarpam/io.hpp"
#include "libarpam/signal.hpp"

namespace arpam::recon {

namespace pipeline {

struct Pipeline {
  explicit Pipeline(size_t n);
  void operator()(const Eigen::ArrayXd &input, Eigen::ArrayXd &output);
  void operator()(const Eigen::MatrixXd &input, Eigen::MatrixXd &output);
};

class FIRFilter : public Pipeline {
public:
  FIRFilter(size_t n, int numtaps, const Eigen::ArrayXd &freq,
            const Eigen::ArrayXd &gain)
      : Pipeline(n), kernel(signal::firwin2(numtaps, freq, gain)) {}

  void operator()(const Eigen::ArrayXd &input, Eigen::ArrayXd &output) {}
  void operator()(const Eigen::MatrixXd &input, Eigen::MatrixXd &output) {}

private:
  Eigen::ArrayXd kernel;
};

} // namespace pipeline
/*
Filt
*/
inline void recon(const Eigen::MatrixXd &rf, Eigen::MatrixXd &output) {

  if (rf.cols() != output.cols() || rf.rows() != output.rows()) {
    throw std::invalid_argument("Sizes of rf and output are different.");
  }

  // Get FIR filter kernel
  const int numtaps = 65;
  Eigen::ArrayXd freq(4);
  freq << 0., 0.1, 0.3, 1.;
  Eigen::ArrayXd gain(4);
  gain << 0., 1., 1., 0.;
  const auto kernel = arpam::signal::firwin2(numtaps, freq, gain);

  const auto aline_size = rf.rows();
  const auto num_alines = rf.cols();

  // Env
  signal::EnvelopDetection envelope_detection(aline_size);

  // Create temp buffers
  Eigen::ArrayXd rf_filt(aline_size);
  const auto conv_offset = numtaps / 2 - (numtaps % 2 == 0);

  Eigen::ArrayXd rf_env(aline_size);

  // For every Aline
  for (auto col_i = 0; col_i < rf.cols(); col_i++) {
    const Eigen::ArrayXd rf_aline = rf.col(col_i);
    rf_filt.setZero();

    // FIR filter
    fftconv::oaconvolve_fftw_same(
        std::span<const double>(rf_aline.data(), rf_aline.size()),
        std::span<const double>(kernel.data(), kernel.size()),
        std::span<double>(rf_filt.data(), rf_filt.size()));

    // Envelope detection
    envelope_detection(rf_filt, rf_env);

    output.col(col_i) = rf_env;
  }
};

} // namespace arpam::recon
