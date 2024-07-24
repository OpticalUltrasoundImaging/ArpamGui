#pragma once

#include <numbers>

namespace uspam::beamformer {

template <typename T> T deg2rad(T deg) {
  constexpr T ONE_EIGHTY = 180.0;
  return deg * std::numbers::pi_v<T> / ONE_EIGHTY;
}
} // namespace uspam::beamformer