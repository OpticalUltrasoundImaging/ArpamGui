#pragma once

#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/beamformer/common.hpp"
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
  [[nodiscard]] Float dr() const { return vs * dt * 1e3 / PAUS; }

  // Transducer focus angle
  [[nodiscard]] Float angle() const {
    return std::asin(static_cast<Float>(d / (2 * f)));
  }

  [[nodiscard]] Float illumAngleRad() const {
    return static_cast<Float>(deg2rad(illumAngleDeg));
  }

  static auto make_PA(int numAlinesPerBScan = 1000) {
    // NOLINTBEGIN(*-magic-numbers)
    const SaftDelayParams<Float> saftParams{
        static_cast<Float>(6.2),
        static_cast<Float>(1.5e3),
        static_cast<Float>(1.0 / 180e6),
        static_cast<Float>(2 * std::numbers::pi / numAlinesPerBScan),
        static_cast<Float>(15.0),
        static_cast<Float>(8.5),
        static_cast<Float>(5.0),
    };
    // NOLINTEND(*-magic-numbers)
    return saftParams;
  }

  static auto make_US(int numAlinesPerBscan = 1000) {
    auto params = make_PA(numAlinesPerBscan);
    params.PAUS = 2;
    return params;
  }
};

template <Floating Float>
[[nodiscard]] auto computeSaftTimeDelay(const SaftDelayParams<Float> &p,
                                        int zStart = -1, int zEnd = -1) {
  // [pts] z start and end points of SAFT.
  // By default start = (half focal distance), end = (1.5x focal distance)
  constexpr auto pi = std::numbers::pi_v<Float>;

  // where SAFT should start (as a fraction of focal length)
  constexpr Float SAFT_START = 0.25;
  constexpr Float SAFT_END = 1.5;

  if (zStart < 0) {
    zStart = static_cast<int>(std::round((p.f * SAFT_START) / p.dr()));
  }

  if (zEnd < 0) {
    zEnd = static_cast<int>(std::round((p.f * SAFT_END) / p.dr()));
  }

  const int max_saft_lines = 15;

  // number of saft lines as a func of z
  arma::Col<uint8_t> nLines(zEnd - zStart, arma::fill::zeros);
  arma::Mat<Float> timeDelay(zEnd - zStart, max_saft_lines, arma::fill::zeros);

  for (int j = 1; j < max_saft_lines; ++j) {
    // relative position to the transducer center dr2 and ang2
    const auto ang1 = j * p.da;

    for (int i = 0; i < zEnd - zStart; ++i) {
      const auto dr1 = (i + zStart) * p.dr();
      const auto r = p.rt + (i + zStart) * p.dr();

      const auto dr2 =
          std::sqrt(r * r + p.rt * p.rt - 2 * r * p.rt * std::cos(ang1));
      const auto ang2 =
          pi - std::acos((p.rt * p.rt + dr2 * dr2 - r * r) / (2 * p.rt * dr2));

      // Determine if point is within the light beam field
      if (ang2 >= p.illumAngleRad()) {
        continue;
      }

      // Determine if point is within the transducer field

      // distance to focus
      const auto dr3 =
          std::sqrt(p.f * p.f + dr2 * dr2 - 2 * p.f * dr2 * std::cos(ang2));

      // angle wrt focal line
      const auto ang3 =
          std::acos((p.f * p.f + dr3 * dr3 - dr2 * dr2) / (2 * p.f * dr3));

      if (dr3 <= p.f && ang3 <= p.angle()) {
        timeDelay.at(i, j) = (abs(p.f - dr1) - dr3) / p.dr();
        nLines.at(i) += 1;
      } else if ((pi - ang3) <= p.angle()) {
        timeDelay.at(i, j) = (dr3 - abs(p.f - dr1)) / p.dr();
        nLines.at(i) += 1;
      }
    }
  }

  return TimeDelay<Float>{timeDelay, nLines, zStart, zEnd};
}

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft(const TimeDelay<Float> &timeDelay,
                            const arma::Mat<Float> &rf);

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft_v2(const TimeDelay<Float> &timeDelay,
                               const arma::Mat<Float> &rf,
                               size_t truncated = 0);

} // namespace uspam::beamformer
