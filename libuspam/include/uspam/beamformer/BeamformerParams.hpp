#pragma once

#include <cmath>
#include <numbers>
#include <uspam/beamformer/common.hpp>

namespace uspam::beamformer {

/**
 * @brief SAFT parameters relating to transducer geometry, rotation geometry,
 * and illumination geometry
 */
template <typename Float> struct BeamformerParams {
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
    const BeamformerParams<Float> saftParams{
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

} // namespace uspam::beamformer