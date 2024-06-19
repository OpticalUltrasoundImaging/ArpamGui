#pragma once

#include "fftconv.hpp"
#include "uspam/imutil.hpp"
#include "uspam/io.hpp"
#include "uspam/reconParams.hpp"
#include "uspam/signal.hpp"
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace uspam::recon {

using fftconv::FloatOrDouble;

void recon(const arma::mat &rf, const arma::vec &kernel, arma::mat &env);

template <typename T>
auto logCompress(T val, T noiseFloor, T desiredDynamicRangeDB) {
  // NOLINTNEXTLINE(*-magic-numbers)
  T compressedVal = 20.0 * std::log10(val / noiseFloor);
  compressedVal = std::max(compressedVal, T(0));
  compressedVal = std::min(compressedVal, desiredDynamicRangeDB);
  return compressedVal / desiredDynamicRangeDB;
}

// NOLINTBEGIN(*-magic-numbers)
template <FloatOrDouble Tin, typename Tout> Tin logCompressFct();
template <> inline consteval double logCompressFct<double, double>() {
  return 1.0;
}
template <> inline consteval float logCompressFct<float, float>() {
  return 1.0F;
}
template <> inline consteval double logCompressFct<double, uint8_t>() {
  return 255.0;
}
template <> inline consteval float logCompressFct<float, uint8_t>() {
  return 255.0;
}
// NOLINTEND(*-magic-numbers)

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
            logCompress(val, noiseFloor, desiredDynamicRangeDB) *
            logCompressFct<T, Tout>();

        if constexpr (std::is_same_v<T, Tout>) {
          xLog(i, j) = compressedVal;
        } else {
          xLog(i, j) = static_cast<Tout>(compressedVal);
        }
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

// FIR filter + Envelope detection + log compression
void reconOneScan(const ReconParams2 &params, io::PAUSpair<double> &rf,
                  io::PAUSpair<uint8_t> &rfLog, bool flip = false);

[[nodiscard]] auto reconOneScan(const ReconParams2 &params,
                                io::PAUSpair<double> &rf, bool flip = false)
    -> io::PAUSpair<uint8_t>;

void reconOneScan(const ReconParams &params, arma::Mat<double> &rf,
                  arma::Mat<uint8_t> &rfLog, bool flip = false);

} // namespace uspam::recon
