#pragma once
#include <armadillo>
#include <cmath>
#include <numbers>
#include <tuple>
#include <uspam/fft.hpp>
#include <uspam/signal.hpp>

namespace uspam::saft {

inline double deg2rad(double deg) { return deg * std::numbers::pi / 180.0; }

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

  FloatType f;     // [mm] transducer focal length
  FloatType d;     // [mm] transducer diameter
  FloatType angle; // [rad] transducer focus angle

  FloatType angleLight; // [rad] illumination angle

  // [mm] spatial step size
  [[nodiscard]] FloatType dr() const { return vs * dt * 1e3; }

  static auto make() {
    // NOLINTBEGIN(*-magic-numbers)
    SaftDelayParams saftParams{
        6.2,
        1.5e3,
        1.0 / 180e6,
        2 * std::numbers::pi / 1000,
        15.0,
        8.5,
        std::asin(static_cast<FloatType>(8.5 / (2 * 15.0))),
        static_cast<float>(deg2rad(5.0)),
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
  const auto pi = std::numbers::pi_v<FloatType>;

  if (zStart < 0) {
    zStart = static_cast<int>(std::round((p.f * 0.25) / p.dr()));
  }

  if (zEnd < 0) {
    zEnd = static_cast<int>(std::round((p.f * 1.5) / p.dr()));
  }

  const int max_saft_lines = 15;

  // number of saft lines as a func of z
  arma::Col<uint8_t> nLines(zEnd - zStart, arma::fill::zeros);
  arma::Mat<FloatType> timeDelay(zEnd - zStart, max_saft_lines,
                                 arma::fill::zeros);

  for (int j = 1; j < max_saft_lines; ++j) {
    // relative position to the transducer center dr2 and ang2

    for (int i = zStart; i < zEnd; ++i) {

      const auto ang1 = j * p.da;

      const auto dr1 = i * p.dr();
      const auto r = p.rt + i * p.dr();

      const auto dr2 =
          std::sqrt(r * r + p.rt * p.rt - 2 * r * p.rt * std::cos(ang1));
      const auto ang2 =
          pi - std::acos((p.rt * p.rt + dr2 * dr2 - r * r) / (2 * p.rt * dr2));

      // Determine if point is within the light beam field
      if (ang2 >= p.angleLight) {
        continue;
      }

      // Determine if point is within the transducer field

      // distance to focus
      const auto dr3 =
          std::sqrt(p.f * p.f + dr2 * dr2 - 2 * p.f * dr2 * std::cos(ang2));

      // angle wrt focal line
      const auto ang3 =
          std::acos((p.f * p.f + dr3 * dr3 - dr2 * dr2) / (2 * p.f * dr3));

      if (dr3 <= p.f && ang3 <= p.angle) {
        timeDelay(i - zStart, j) = (abs(p.f - dr1) - dr3) / p.dr();
        nLines(i - zStart) += 1;
      } else if ((pi - ang3) <= p.angle) {
        timeDelay(i - zStart, j) = (dr3 - abs(p.f - dr1)) / p.dr();
        nLines(i - zStart) += 1;
      }
    }
  }

  return TimeDelay<FloatType>{timeDelay, nLines, zStart, zEnd};
}

template <typename RfType, Floating FloatType>
auto apply_saft(const TimeDelay<FloatType> &timeDelay,
                const arma::Mat<RfType> &rf) {
  const int nScans = rf.n_cols;
  const int nPts = rf.n_rows;

  arma::Mat<RfType> rf_saft = rf; // copy
  arma::Mat<uint8_t> n_saft(rf.n_rows, rf.n_cols, arma::fill::ones);

  arma::Mat<FloatType> CF_denom = arma::square(rf);

  for (int j = 0; j < nScans; ++j) {
    for (int iz = timeDelay.zStart; iz < timeDelay.zEnd; ++iz) {
      for (int dj_saft = 0;
           dj_saft < timeDelay.saftLines(iz - timeDelay.zStart); ++dj_saft) {

        const int iz_delayed = static_cast<int>(std::round(
            iz + timeDelay.timeDelay(iz - timeDelay.zStart, dj_saft)));

        if (iz_delayed >= nPts) {
          continue;
        }

        const auto val = rf(iz_delayed, j);

        {
          const auto j_saft = (j - dj_saft + nScans) % nScans;
          rf_saft(iz, j_saft) += val;
          CF_denom(iz, j_saft) += val * val;
          n_saft(iz, j_saft) += 1;
        }

        {
          const auto j_saft = (j + dj_saft + nScans) % nScans;
          rf_saft(iz, j_saft) += val;
          CF_denom(iz, j_saft) += val * val;
          n_saft(iz, j_saft) += 1;
        }
      }
    }
  }

  // CF = PA_saft ** 2 / (CF_denom * n_saft)
  arma::Mat<FloatType> CF(rf_saft.n_rows, rf_saft.n_cols, arma::fill::zeros);

  // rf_saft_cf = rf_saft * CF / n_saft
  arma::Mat<FloatType> rf_saft_cf(rf_saft.n_rows, rf_saft.n_cols,
                                  arma::fill::zeros);

  for (int col = 0; col < rf_saft.n_cols; ++col) {
    for (int row = 0; row < rf_saft.n_rows; ++row) {
      const auto nom = rf_saft(row, col) * rf_saft(row, col);
      const auto denom = CF_denom(row, col) * n_saft(row, col);

      if (denom != 0) {
        CF(row, col) = nom / denom;
      } else {
        CF(row, col) = 1;
      }

      rf_saft_cf(row, col) =
          rf_saft(row, col) * CF(row, col) / n_saft(row, col);
    }
  }

  return std::tuple(rf_saft, rf_saft_cf);
}

} // namespace uspam::saft
