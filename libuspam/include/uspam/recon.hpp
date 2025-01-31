#pragma once

#include "uspam/filter.hpp"
#include "uspam/imutil.hpp"
#include "uspam/ioParams.hpp"
#include "uspam/reconParams.hpp"
#include "uspam/signal.hpp"
#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <fftconv/aligned_vector.hpp>
#include <fftconv/fftconv.hpp>
#include <fftconv/hilbert.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace fs = std::filesystem;

namespace uspam::recon {

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
template <fftconv::Floating Tin, typename Tout> Tin logCompressFct();
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
template <fftconv::Floating T, typename Tout>
void logCompress_par(const arma::Mat<T> &x, arma::Mat<Tout> &xLog,
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

template <fftconv::Floating T, typename Tout>
void logCompress(const std::span<const T> x, const std::span<Tout> xLog,
                 const T noiseFloor, const T desiredDynamicRangeDB = 45.0) {
  assert(!x.empty());
  assert(x.size() == xLog.size());

  // Apply log compression with clipping in a single pass
  std::transform(x.begin(), x.end(), xLog.begin(), [&](const T val) {
    if constexpr (std::is_same_v<T, Tout>) {
      return logCompress(val, noiseFloor, desiredDynamicRangeDB);
    }
    return logCompress(val, noiseFloor, desiredDynamicRangeDB) *
           logCompressFct<T, Tout>();
  });
}

// Determine the dynamic range (dB) for a given signal with a known noiseFloor
template <fftconv::Floating T>
auto calcDynamicRange(const std::span<const T> x, const T noiseFloor) {
  // Determine the peak signal value.
  const T peakLevel = *std::max_element(x.begin(), x.end());
  const T dynamicRangeDB = 20.0 * std::log10(peakLevel / noiseFloor);
  return dynamicRangeDB;
}

template <typename T>
using FilterT = std::variant<FIRFilter<T>, ButterworthFilter<T>>;

template <fftconv::Floating T>
auto createFilter(const ReconParams &params) -> FilterT<T> {
  switch (params.filterType) {
  case FilterType::FIR:
    return FIRFilter<T>(params.firTaps, params.bpLowFreq, params.bpHighFreq);
  case FilterType::IIR:
    return ButterworthFilter<T>(params.iirOrder, params.bpLowFreq,
                                params.bpHighFreq);
  default:
    throw std::invalid_argument("Unknown filter type");
  }
}

template <fftconv::Floating T, typename Tlog = uint8_t>
void filterAndEnvelope(const arma::Mat<T> &rfBeamformed, arma::Mat<T> &rfEnv,
                       arma::Mat<Tlog> &rfLog, size_t N, size_t truncate,
                       const ReconParams &params) {

  const auto filter = createFilter<T>(params);

  const cv::Range range(0, static_cast<int>(rfBeamformed.n_cols));
  cv::parallel_for_(range, [&](const cv::Range &range) {
    fftconv::AlignedVector<T> filterBuffer(N);
    constexpr T mV_to_V = 1.0 / 1000;
    const T noiseFloor_V = params.noiseFloor_mV * mV_to_V;

    for (int i = range.start; i < range.end; ++i) {
      const std::span rfCol{rfBeamformed.colptr(i), N};
      const std::span envCol{rfEnv.colptr(i), N};
      const std::span logCol{rfLog.colptr(i) + truncate, N};

      std::visit(
          [&](const auto &filter) { filter.forward(rfCol, filterBuffer); },
          filter);

      fftconv::hilbert<T>(filterBuffer, envCol);
      uspam::recon::logCompress<T, Tlog>(envCol, logCol, noiseFloor_V,
                                         params.desiredDynamicRange);
    }
  });
}

template <typename T>
arma::Mat<T> preprocessRF(const arma::Mat<T> &rf, size_t truncate, bool flip,
                          const uspam::SystemParams &system) {
  arma::Mat<T> truncatedRF =
      rf.submat(truncate, 0, rf.n_rows - 1, rf.n_cols - 1);

  // Preprocessing (flip, rotate)
  if (flip) {
    // Flip
    uspam::imutil::fliplr_inplace(truncatedRF);

    // Rotate
    if (system.rotateOffset != 0) {
      truncatedRF = arma::shift(truncatedRF, system.rotateOffset, 1);
    }
  }

  return truncatedRF;
}

} // namespace uspam::recon
