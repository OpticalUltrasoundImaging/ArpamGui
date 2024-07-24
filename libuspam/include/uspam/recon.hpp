#pragma once

#include "fftconv.hpp"
#include "uspam/imutil.hpp"
#include "uspam/ioParams.hpp"
#include "uspam/reconParams.hpp"
#include "uspam/signal.hpp"
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <type_traits>

namespace fs = std::filesystem;

namespace uspam::recon {

template <Floating T>
void recon(const arma::Mat<T> &rf, const arma::Col<T> &kernel,
           arma::Mat<T> &env) {
  const cv::Range range(0, static_cast<int>(rf.n_cols));
  // cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
  arma::Col<T> rf_filt(rf.n_rows);
  for (int i = range.start; i < range.end; ++i) {
    const auto src = rf.unsafe_col(i);
    auto dst = env.unsafe_col(i);
    fftconv::oaconvolve_fftw_same<T>(src, kernel, rf_filt);
    signal::hilbert_abs_r2c<T>(rf_filt, dst);
  }
  // });
}

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <Arithmetic T>
auto logCompress(T val, T noiseFloor, T desiredDynamicRangeDB) {
  // NOLINTNEXTLINE(*-magic-numbers)
  T compressedVal = 20.0 * std::log10(val / noiseFloor);
  compressedVal = std::max(compressedVal, T(0));
  compressedVal = std::min(compressedVal, desiredDynamicRangeDB);
  return compressedVal / desiredDynamicRangeDB;
}

// NOLINTBEGIN(*-magic-numbers)
template <Floating Tin, typename Tout> Tin logCompressFct();
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
template <Floating T, typename Tout>
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

        xLog(i, j) = static_cast<Tout>(compressedVal);
      }
    }
    //}(cv::Range(0, x.n_cols));
  });
}

template <Floating T>
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
template <Floating T>
auto calcDynamicRange(const std::span<const T> x, const T noiseFloor) {
  // Determine the peak signal value.
  const T peakLevel = *std::max_element(x.begin(), x.end());
  const T dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);
  return dynamicRangeDB;
}

// FIR filter + Envelope detection + log compression for PA/US pair
// rf contains the RF signal, and results are saved to rfLog
template <Floating T>
void reconOneScan(const ReconParams2 &params, io::PAUSpair<T> &rf,
                  io::PAUSpair<T> &rfBeamformed, io::PAUSpair<uint8_t> &rfLog,
                  bool flip) {
  reconOneScan<T>(params.PA, rf.PA, rfBeamformed.PA, rfLog.PA, flip);
  reconOneScan<T>(params.US, rf.US, rfBeamformed.US, rfLog.US, flip);
}

// Beamform + FIR filter + Envelope detection + log compression for one
template <Floating T>
void reconOneScan(const ReconParams &params, arma::Mat<T> &rf,
                  arma::Mat<T> &rfBeamformed, arma::Mat<T> &rfEnv,
                  arma::Mat<uint8_t> &rfLog, bool flip) {
  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf);

    // Do rotate
    rf = arma::shift(rf, params.rotateOffset, 1);
  }

  // Truncate the pulser/laser artifact
  rf.head_rows(params.truncate - 1).zeros();

  // Beamform
  beamform(rf, rfBeamformed, params.beamformerType);

  // compute filter kernels
  const auto kernel = [&] {
    constexpr int numtaps = 95;
    if constexpr (std::is_same_v<T, double>) {
      return signal::firwin2<double>(numtaps, params.filterFreq,
                                     params.filterGain);
    } else {
      const auto _kernel = signal::firwin2<double>(numtaps, params.filterFreq,
                                                   params.filterGain);
      const auto kernel = arma::conv_to<arma::Col<T>>::from(_kernel);
      return kernel;
    }
  }();

  if (rf.n_rows != rfEnv.n_rows || rf.n_cols != rfEnv.n_cols) {
    rfEnv.set_size(rf.n_rows, rf.n_cols);
  }

  recon<T>(rfBeamformed, kernel, rfEnv);

  constexpr float fct_mV2V = 1.0F / 1000;
  rfLog.set_size(rf.n_rows, rf.n_cols);
  logCompress<T>(rfEnv, rfLog, params.noiseFloor_mV * fct_mV2V,
                 params.desiredDynamicRange);
}

// FIR filter + Envelope detection + log compression for one
template <Floating T>
void reconOneScan(const ReconParams &params, arma::Mat<T> &rf,
                  arma::Mat<T> &rfEnv, arma::Mat<uint8_t> &rfLog, bool flip) {
  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf);

    // Do rotate
    rf = arma::shift(rf, params.rotateOffset, 1);
  }

  // Truncate the pulser/laser artifact
  rf.head_rows(params.truncate - 1).zeros();

  // compute filter kernels
  const auto kernel = [&] {
    constexpr int numtaps = 95;
    if constexpr (std::is_same_v<T, double>) {
      return signal::firwin2<double>(numtaps, params.filterFreq,
                                     params.filterGain);
    } else {
      const auto _kernel = signal::firwin2<double>(numtaps, params.filterFreq,
                                                   params.filterGain);
      const auto kernel = arma::conv_to<arma::Col<T>>::from(_kernel);
      return kernel;
    }
  }();

  if (rf.n_rows != rfEnv.n_rows || rf.n_cols != rfEnv.n_cols) {
    rfEnv.set_size(rf.n_rows, rf.n_cols);
  }

  recon<T>(rf, kernel, rfEnv);

  constexpr float fct_mV2V = 1.0F / 1000;
  rfLog.set_size(rf.n_rows, rf.n_cols);
  logCompress<T>(rfEnv, rfLog, params.noiseFloor_mV * fct_mV2V,
                 params.desiredDynamicRange);
}
} // namespace uspam::recon
