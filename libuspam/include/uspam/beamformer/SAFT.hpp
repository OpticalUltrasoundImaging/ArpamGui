#pragma once

#include <armadillo>
#include <cmath>
#include <numbers>
#include <opencv2/opencv.hpp>
#include <uspam/SystemParams.hpp>
#include <uspam/beamformer/BeamformerType.hpp>
#include <uspam/fft.hpp>
#include <uspam/signal.hpp>

namespace uspam::beamformer {

template <Floating Float> struct TimeDelay {
  arma::Mat<Float> timeDelay;
  arma::Col<uint8_t> saftLines;
  int zStart{};
  int zEnd{};
};

[[nodiscard]] inline auto computeSaftTimeDelay(const SystemParams &p,
                                               int zStart = -1, int zEnd = -1) {
  // [pts] z start and end points of SAFT.
  // By default start = (half focal distance), end = (1.5x focal distance)
  constexpr auto pi = std::numbers::pi_v<float>;

  // where SAFT should start (as a fraction of focal length)
  constexpr float SAFT_START = 0.25F;
  constexpr float SAFT_END = 1.5F;

  const auto dr = p.dr();
  const auto dr_inv = 1 / dr;
  const auto mm2pix = [dr_inv](float mm) -> int {
    return static_cast<int>(std::round(mm * dr_inv));
  };

  if (zStart < 0) {
    zStart = mm2pix(p.focalLength * SAFT_START);
  }

  if (zEnd < 0) {
    zEnd = mm2pix(p.focalLength * SAFT_END);
  }

  const int max_saft_lines = 12;

  // number of saft lines as a func of z
  arma::Col<uint8_t> nLines(zEnd - zStart, arma::fill::zeros);
  arma::Mat<float> timeDelay(zEnd - zStart, max_saft_lines, arma::fill::zeros);

  const auto f = mm2pix(p.focalLength);
  const auto rt = mm2pix(p.transducerOffset);
  const auto da = p.da();

  for (int j = 1; j < max_saft_lines; ++j) {
    // relative position to the transducer center dr2 and ang2
    const auto ang1 = j * da;

    for (int i = 0; i < zEnd - zStart; ++i) {
      const auto dr1 = i + zStart;
      const auto r = rt + dr1;

      const auto dr2 = std::sqrt(r * r + rt * rt - 2 * r * rt * std::cos(ang1));
      const auto ang2 =
          pi - std::acos((rt * rt + dr2 * dr2 - r * r) / (2 * rt * dr2));

      // Determine if point is within the light beam field
      if (ang2 >= p.illumAngleRad()) {
        continue;
      }

      // Determine if point is within the transducer field

      // distance to focus
      const auto dr3 =
          std::sqrt(f * f + dr2 * dr2 - 2 * f * dr2 * std::cos(ang2));

      // angle wrt focal line
      const auto ang3 =
          std::acos((f * f + dr3 * dr3 - dr2 * dr2) / (2 * f * dr3));

      if (dr3 <= f && ang3 <= p.focalAngle()) {
        const auto delayVal = (abs(f - dr1) - dr3);
        timeDelay.at(i, j) = delayVal * p.saftTimeDelayMultiplier;
        nLines.at(i) += 1;
      } else if ((pi - ang3) <= p.focalAngle()) {
        const auto delayVal = (dr3 - abs(f - dr1));
        timeDelay.at(i, j) = delayVal * p.saftTimeDelayMultiplier;
        nLines.at(i) += 1;
      }
    }
  }

  return TimeDelay<float>{timeDelay, nLines, zStart, zEnd};
}

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft(const TimeDelay<Float> &timeDelay,
                            const arma::Mat<Float> &rf);

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft_v2(const TimeDelay<Float> &timeDelay,
                               const arma::Mat<Float> &rf,
                               size_t truncated = 0);

} // namespace uspam::beamformer
