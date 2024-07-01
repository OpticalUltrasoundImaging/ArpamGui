#pragma once

#include <numbers>

namespace uspam::beamformer {

inline double deg2rad(double deg) { return deg * std::numbers::pi / 180.0; }
} // namespace uspam::beamformer