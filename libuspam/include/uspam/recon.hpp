#pragma once

#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "fftconv.hpp"
#include "uspam/imutil.hpp"
#include "uspam/io.hpp"
#include "uspam/signal.hpp"

#include <opencv2/opencv.hpp>

namespace uspam::recon {

using fftconv::FloatOrDouble;

void recon(const arma::mat &rf, const arma::vec &kernel, arma::mat &env);

// Log compress to range of 0 - 1
template <FloatOrDouble T>
auto logCompress(const arma::Mat<T> &x, arma::Mat<T> &xLog, const T noiseFloor,
                 const T desiredDynamicRangeDB = 45.0) -> T {
  assert(!x.empty());
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
        compressedValue = std::max(compressedValue, T(0));
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
  assert(!x.empty());
  assert(x.size() == xLog.size());

  // Determine the peak signal value.
  const T peakLevel = *std::max_element(x.begin(), x.end());
  const T dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);

  // Apply log compression with clipping in a single pass
  std::transform(x.begin(), x.end(), xLog.begin(), [&](const T val) {
    T compressedValue = 20.0 * std::log10(val / noiseFloor);
    compressedValue = std::max(compressedValue, T(0));
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
                    bool flip = false) const;

  [[nodiscard]] auto reconOneScan(io::PAUSpair<double> &rf,
                                  bool flip = false) const
      -> io::PAUSpair<double>;
};

} // namespace uspam::recon
