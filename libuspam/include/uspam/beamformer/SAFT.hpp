#pragma once

#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/beamformer/common.hpp"
#include "uspam/ioParams.hpp"
#include <fmt/format.h>
// #include "uspam/reconParams.hpp"
#include <armadillo>
#include <cmath>
#include <numbers>
#include <opencv2/opencv.hpp>
#include <uspam/fft.hpp>
#include <uspam/signal.hpp>

namespace uspam::beamformer {

template <Floating Float> struct TimeDelay {
  arma::Mat<Float> timeDelay;
  arma::Col<uint8_t> saftLines;
  int zStart{};
  int zEnd{};
};

/**
 * @brief SAFT parameters relating to transducer geometry, rotation geometry,
 * and illumination geometry
 */
template <Floating Float> struct SaftDelayParams {
  Float rt; // [mm] distance from axis of rotation to transducer surface
  Float vs; // [m/s] sound speed
  Float dt; // [s] timestep
  Float da; // [rad] angle step size in each rotation

  Float f; // [mm] transducer focal length
  Float d; // [mm] transducer diameter

  Float illumAngleDeg; // [deg] illumination angle

  Float PAUS = 1; // 1 trip for PA, 2 trips for US

  // [mm] spatial step size
  // NOLINTNEXTLINE(*-magic-numbers)
  [[nodiscard]] constexpr Float dr() const { return vs * dt * 1e3 / PAUS; }

  // Transducer focus angle
  [[nodiscard]] constexpr Float angle() const {
    return std::asin(static_cast<Float>(d / (2 * f)));
  }

  [[nodiscard]] constexpr Float illumAngleRad() const {
    return static_cast<Float>(deg2rad(illumAngleDeg));
  }

  // Need to solve circular includes
  // static constexpr auto make(uspam::recon::ReconParams reconParams,
  //                            uspam::io::IOParams ioparams) {
  //   // NOLINTBEGIN(*-magic-numbers)

  //   return SaftDelayParams<Float>{
  //       .rt = 6.2,
  //       .vs = reconParams.SoS,
  //       .dt = static_cast<Float>(1.0 / reconParams.fs),
  //       .da = 2 * static_cast<Float>(std::numbers::pi) /
  //             static_cast<Float>(ioparams.alinesPerBscan),
  //       .f = 15.0,
  //       .d = 8.5,
  //       .illumAngleDeg = 5.0,
  //       .PAUS = reconParams.isPA ? 1 : 2,
  //   };
  //   // NOLINTEND(*-magic-numbers)
  // }

  static constexpr auto make_PA(int numAlinesPerBScan = 1000,
                                Float fs = 180e6) {
    // NOLINTBEGIN(*-magic-numbers)
    return SaftDelayParams<Float>{
        .rt = 6.2,
        .vs = 1.5e3,
        .dt = static_cast<Float>(1.0 / fs),
        .da = 2 * static_cast<Float>(std::numbers::pi) /
              static_cast<Float>(numAlinesPerBScan),
        .f = 15.0,
        .d = 8.5,
        .illumAngleDeg = 5.0,
    };
    // NOLINTEND(*-magic-numbers)
  }

  static constexpr auto make_US(int numAlinesPerBscan = 1000,
                                Float fs = 180e6) {
    auto params = make_PA(numAlinesPerBscan, fs);
    params.PAUS = 2;
    return params;
  }
};

template <Floating Float>
[[nodiscard]] TimeDelay<Float>
computeSaftTimeDelay(const SaftDelayParams<Float> &p, int zStart = -1,
                     int zEnd = -1);

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft(const TimeDelay<Float> &timeDelay,
                            const arma::Mat<Float> &rf);

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft_v2(const TimeDelay<Float> &timeDelay,
                               const arma::Mat<Float> &rf, int offset = 0);

} // namespace uspam::beamformer
