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

template <Floating FloatType> struct TimeDelay {
  arma::Mat<FloatType> timeDelay;
  arma::Col<uint8_t> saftLines;
  int zStart{};
  int zEnd{};
};

/**
 * @brief SAFT parameters relating to transducer geometry, rotation geometry,
 * and illumination geometry
 */
template <Floating FloatType> struct SaftDelayParams {
  FloatType rt; // [mm] distance from axis of rotation to transducer surface
  FloatType vs; // [m/s] sound speed
  FloatType dt; // [s] timestep
  FloatType da; // [rad] angle step size in each rotation

  FloatType f; // [mm] transducer focal length
  FloatType d; // [mm] transducer diameter

  FloatType illumAngleDeg; // [deg] illumination angle

  // [mm] spatial step size
  // NOLINTNEXTLINE(*-magic-numbers)
  [[nodiscard]] FloatType dr() const { return vs * dt * 1e3; }

  // Transducer focus angle
  [[nodiscard]] FloatType angle() const {
    return std::asin(static_cast<FloatType>(d / (2 * f)));
  }

  [[nodiscard]] FloatType illumAngleRad() const {
    return static_cast<FloatType>(deg2rad(illumAngleDeg));
  }

  static auto make() {
    // NOLINTBEGIN(*-magic-numbers)
    SaftDelayParams<FloatType> saftParams{
        6.2, 1.5e3, 1.0 / 180e6, 2 * std::numbers::pi / 1000, 15.0, 8.5, 5.0,
    };
    // NOLINTEND(*-magic-numbers)
    return saftParams;
  }
};

template <Floating FloatType>
[[nodiscard]] auto computeSaftTimeDelay(const SaftDelayParams<FloatType> &p,
                                        int zStart = -1, int zEnd = -1) {
  // [pts] z start and end points of SAFT.
  // By default start = (half focal distance), end = (1.5x focal distance)
  constexpr auto pi = std::numbers::pi_v<FloatType>;

  // where SAFT should start (as a fraction of focal length)
  constexpr FloatType SAFT_START = 0.25;
  constexpr FloatType SAFT_END = 1.5;

  if (zStart < 0) {
    zStart = static_cast<int>(std::round((p.f * SAFT_START) / p.dr()));
  }

  if (zEnd < 0) {
    zEnd = static_cast<int>(std::round((p.f * SAFT_END) / p.dr()));
  }

  const int max_saft_lines = 15;

  // number of saft lines as a func of z
  arma::Col<uint8_t> nLines(zEnd - zStart, arma::fill::zeros);
  arma::Mat<FloatType> timeDelay(zEnd - zStart, max_saft_lines,
                                 arma::fill::zeros);

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

  return TimeDelay<FloatType>{timeDelay, nLines, zStart, zEnd};
}

template <typename RfType, Floating FloatType, BeamformerType BfType>
auto apply_saft(const TimeDelay<FloatType> &timeDelay,
                const arma::Mat<RfType> &rf) {
  const int nScans = rf.n_cols;
  const int nPts = rf.n_rows;

  arma::Mat<RfType> rf_saft = rf; // copy
  arma::Mat<uint8_t> n_saft(rf.n_rows, rf.n_cols, arma::fill::ones);
  arma::Mat<FloatType> CF_denom = arma::square(rf);

  cv::parallel_for_(
      cv::Range(timeDelay.zStart, timeDelay.zEnd), [&](const cv::Range range) {
        for (int j = 0; j < nScans; ++j) {
          for (int iz = range.start; iz < range.end; ++iz) {
            // for (int iz = timeDelay.zStart; iz < timeDelay.zEnd; ++iz) {

            const auto NLines = timeDelay.saftLines.at(iz - timeDelay.zStart);

            for (int dj_saft = 0; dj_saft < NLines; ++dj_saft) {
              const auto dt =
                  timeDelay.timeDelay.at(iz - timeDelay.zStart, dj_saft);
              const int iz_delayed = static_cast<int>(std::round(iz + dt));

              if (iz_delayed >= nPts) {
                continue;
              }

              const auto val = rf.at(iz_delayed, j);
              const auto valSq = val * val;

              {
                const auto j_saft = (j - dj_saft + nScans) % nScans;
                rf_saft.at(iz, j_saft) += val;
                CF_denom.at(iz, j_saft) += valSq;
                n_saft.at(iz, j_saft) += 1;
              }

              {
                const auto j_saft = (j + dj_saft + nScans) % nScans;
                rf_saft.at(iz, j_saft) += val;
                CF_denom.at(iz, j_saft) += valSq;
                n_saft.at(iz, j_saft) += 1;
              }
            }
          }
        }
      });

  // CF = PA_saft ** 2 / (CF_denom * n_saft)
  // rf_saft_cf = rf_saft * CF / n_saft

  for (int col = 0; col < rf_saft.n_cols; ++col) {
    for (int row = 0; row < rf_saft.n_rows; ++row) {
      const auto rf_saft_ = rf_saft.at(row, col);
      const auto n_saft_ = n_saft.at(row, col);

      if constexpr (BfType == BeamformerType::SAFT) {
        rf_saft.at(row, col) = rf_saft_ / n_saft_;
      } else { // BfType == BeamformerType::SAFT_CF
        const auto nom = rf_saft_ * rf_saft_;
        const auto denom = CF_denom.at(row, col) * n_saft_;

        const auto CF_ = denom != 0 ? nom / denom : 1;
        rf_saft.at(row, col) = rf_saft_ * CF_ / n_saft_;
      }
    }
  }

  return rf_saft;
}

} // namespace uspam::beamformer
