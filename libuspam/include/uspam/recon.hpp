#pragma once

#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "fftconv.hpp"
#include "opencv2/opencv.hpp"
#include "uspam/fft.hpp"
#include "uspam/io.hpp"
#include "uspam/signal.hpp"

#include <opencv2/opencv.hpp>

namespace uspam::recon {

using fftconv::FloatOrDouble;

auto recon(const arma::mat &rf, const arma::vec &kernel, arma::mat &env) {
  // TODO FIR filter

  //cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
  //  arma::vec rf_filt(rf.n_rows);
  //  for (int i = range.start; i < range.end; ++i) {
  //    const auto src = rf.unsafe_col(i);
  //    auto dst = env.unsafe_col(i);
  //    fftconv::oaconvolve_fftw_same<double>(src, kernel, rf_filt);
  //    signal::hilbert_abs(rf_filt, dst);
  //  }
  //});
   arma::vec rf_filt(rf.n_rows);
   for (int i = 0; i < rf.n_cols; ++i) {
     const auto src = rf.unsafe_col(i);
     auto dst = env.unsafe_col(i);
     fftconv::oaconvolve_fftw_same<double>(src, kernel, rf_filt);
     signal::hilbert_abs(rf_filt, dst);
   }
}

template <FloatOrDouble T>
auto logCompress(const arma::Mat<T> &x, arma::Mat<T> &xLog, const T noiseFloor,
                 const T desiredDynamicRangeDB = 45.0) -> T {
  assert(x.size() > 0);
  assert(x.size() == xLog.size());

  // Determine the peak signal value.
  const T peakLevel = *std::max_element(x.begin(), x.end());
  const T dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);

  // Apply log compression with clipping in a single pass
  cv::parallel_for_(cv::Range(0, x.n_cols), [&](const cv::Range &range) {
    for (int j = range.start; j < range.end; ++j) {
      for (int i = 0; i < x.n_rows; ++i) {
        const auto val = x(i, j);
        T compressedValue = 20.0 * std::log10(val / noiseFloor);
        compressedValue = std::max(compressedValue, 0.0);
        compressedValue = std::min(compressedValue, desiredDynamicRangeDB);
        compressedValue /= desiredDynamicRangeDB;

        xLog(i, j) = compressedValue;
      }
    }
    //}(cv::Range(0, x.n_cols));
  });

  return dynamicRangeDB;
}

template <FloatOrDouble T>
auto logCompress(const std::span<const T> x, const std::span<T> xLog,
                 const T noiseFloor, const T desiredDynamicRangeDB = 45.0)
    -> T {
  assert(x.size() > 0);
  assert(x.size() == xLog.size());

  // Determine the peak signal value.
  const T peakLevel = *std::max_element(x.begin(), x.end());
  const T dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);

  // Apply log compression with clipping in a single pass
  std::transform(x.begin(), x.end(), xLog.begin(), [&](const T val) {
    T compressedValue = 20.0 * std::log10(val / noiseFloor);
    compressedValue = std::max(compressedValue, 0.0);
    compressedValue = std::min(compressedValue, desiredDynamicRangeDB);
    return compressedValue / desiredDynamicRangeDB;
  });

  return dynamicRangeDB;
}

// FIR Filter

template <FloatOrDouble T> struct FIRFilterParams {
  int numtaps;
  arma::Col<T> freq;
  arma::Col<T> gain;

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

struct ReconParams2 {
  std::vector<double> filter_freq_PA;
  std::vector<double> filter_gain_PA;
  std::vector<double> filter_freq_US;
  std::vector<double> filter_gain_US;

  int noise_floor_PA;
  int noise_floor_US;

  int desired_dynamic_range_PA;
  int desired_dynamic_range_US;

  int aline_rotation_offset;

  static inline ReconParams2 system2024v1() {
    return ReconParams2{{0, 0.03, 0.035, 0.2, 0.22, 1},
                        {0, 0, 1, 1, 0, 0},
                        {0, 0.1, 0.3, 1},
                        {0, 1, 1, 0},
                        250,
                        200,
                        40,
                        48,
                        22};
  }

  // FIR filter + Envelope detection + log compression
  void reconOneScan(io::PAUSpair<double> &rf, io::PAUSpair<double> &rfLog,
                    bool flip = false) const {

    if (flip) {
      // Do flip
      rf.PA = arma::fliplr(rf.PA);
      rf.US = arma::fliplr(rf.US);

      // Do rotate
      const auto rotate_offset = this->aline_rotation_offset;
      rf.PA = arma::shift(rf.PA, rotate_offset, 1);
      rf.US = arma::shift(rf.US, rotate_offset, 1);
    }

    // compute filter kernels
    const auto kernelPA = signal::firwin2(65, filter_freq_PA, filter_gain_PA);
    const auto kernelUS = signal::firwin2(65, filter_freq_US, filter_gain_US);

    auto env = io::PAUSpair<double>::empty_like(rf);

    arma::mat &PAlog = rfLog.PA;
    arma::mat &USlog = rfLog.US;

    recon(rf.PA, kernelPA, env.PA);
    logCompress<double>(env.PA, PAlog, this->noise_floor_PA,
                        this->desired_dynamic_range_PA);

    recon(rf.US, kernelUS, env.US);
    logCompress<double>(env.US, USlog, this->noise_floor_US,
                        this->desired_dynamic_range_US);
  }

  [[nodiscard]] inline auto reconOneScan(io::PAUSpair<double> &rf,
                                         bool flip = false) const {
    auto rfLog = io::PAUSpair<double>::zeros_like(rf);
    reconOneScan(rf, rfLog, flip);
    return rfLog;
  }
};

} // namespace uspam::recon
