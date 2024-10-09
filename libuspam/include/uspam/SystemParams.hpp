#pragma once

#include <cmath>
#include <numbers>

// NOLINTBEGIN(*-magic-numbers)

namespace uspam {

/*
Describe the imaging system parameters
*/
struct SystemParams {
  float soundSpeed{1500.0F}; // [m/s] Speed of sound
  float fs{180.0e6F};        // [Hz] Sampling frequency

  /*
  ARPAM probe head
  */
  float transducerOffset{6.2};   // [mm]
  float focalLength{15.0};       // [mm] transducer focal length
  float transducerDiameter{8.5}; // [mm]
  float illumAngleDeg{5.0};      // [deg] illumination angle

  /*
  Imaging sequence
  */
  bool flipOnEven{};
  int alinesPerBscan{1000};
  int rotateOffset{};

  // [mm] spatial step size
  [[nodiscard]] float dr() const { return soundSpeed / fs * 1000; }

  // [rad] anglular step in each rotation
  [[nodiscard]] float da() const {
    return 2 * std::numbers::pi_v<float> / static_cast<float>(alinesPerBscan);
  }

  // [rad]
  [[nodiscard]] float illumAngleRad() const {
    constexpr float fct = std::numbers::pi_v<float> / 180.0F;
    return illumAngleDeg * fct;
  }

  // [rad]
  [[nodiscard]] float focalAngle() const {
    return std::asin(transducerDiameter / (2 * focalLength));
  }

  // [m]
  [[nodiscard]] auto pixelSpacing() const { return soundSpeed / fs; }

  [[nodiscard]] auto flip(int frameIdx) const {
    const bool even = frameIdx % 2 == 0;
    return flipOnEven ? even : !even;
  }

  // TODO serialize
};

} // namespace uspam

// NOLINTEND(*-magic-numbers)