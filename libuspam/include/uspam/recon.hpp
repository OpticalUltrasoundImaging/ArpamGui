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

template <typename T>
auto logCompress(T val, T noiseFloor, T desiredDynamicRangeDB) {
  T compressedVal = 20.0 * std::log10(val / noiseFloor);
  compressedVal = std::max(compressedVal, T(0));
  compressedVal = std::min(compressedVal, desiredDynamicRangeDB);
  return compressedVal / desiredDynamicRangeDB;
}

// Log compress to range of 0 - 1
template <FloatOrDouble T, typename Tout>
void logCompress(const arma::Mat<T> &x, arma::Mat<Tout> &xLog,
                 const T noiseFloor, const T desiredDynamicRangeDB = 45.0) {
  assert(!x.empty());
  assert(x.size() == xLog.size());

  // Apply log compression with clipping in a single pass
  cv::parallel_for_(cv::Range(0, x.n_cols), [&](const cv::Range &range) {
    for (int j = range.start; j < range.end; ++j) {
      for (int i = 0; i < x.n_rows; ++i) {
        const auto val = x(i, j);
        const auto compressedVal =
            logCompress(val, noiseFloor, desiredDynamicRangeDB);
        if constexpr (std::is_same_v<T, Tout>)
          xLog(i, j) = compressedVal;
        else
          xLog(i, j) = static_cast<Tout>(compressedVal);
      }
    }
    //}(cv::Range(0, x.n_cols));
  });
}

template <FloatOrDouble T>
void logCompress(const std::span<const T> x, const std::span<T> xLog,
                 const T noiseFloor, const T desiredDynamicRangeDB = 45.0) {
  assert(!x.empty());
  assert(x.size() == xLog.size());

  // Apply log compression with clipping in a single pass
  std::transform(x.begin(), x.end(), xLog.begin(), [&](const T val) {
    return logCompress(val, noiseFloor, desiredDynamicRangeDB);
  });
}

// Determine the dynamic range (dB) for a given signal with a known noiseFloor
template <FloatOrDouble T>
auto calcDynamicRange(const std::span<const T> x, const T noiseFloor) {
  // Determine the peak signal value.
  const T peakLevel = *std::max_element(x.begin(), x.end());
  const T dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);
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
