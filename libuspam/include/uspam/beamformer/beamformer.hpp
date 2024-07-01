#pragma once

#include "uspam/beamformer/SAFT.hpp"
#include <type_traits>

namespace uspam::beamformer {

enum class BeamformerType {
  NONE,
  SAFT,
  SAFT_CF,
};

template <typename T>
void beamform(const arma::Mat<T> &rf, arma::Mat<T> &rfBeamformed,
              BeamformerType beamformer)
  requires std::is_floating_point_v<T>
{
  switch (beamformer) {
  case BeamformerType::SAFT: {
    const auto saftParams = uspam::beamformer::SaftDelayParams<T>::make();
    const auto timeDelay =
        uspam::beamformer::computeSaftTimeDelay<T>(saftParams);
    const auto [rfSaft, rfSaftCF] =
        uspam::beamformer::apply_saft<T, T>(timeDelay, rf);

    rfBeamformed = rfSaft;
  } break;

  case BeamformerType::SAFT_CF: {
    const auto saftParams = uspam::beamformer::SaftDelayParams<T>::make();
    const auto timeDelay =
        uspam::beamformer::computeSaftTimeDelay<T>(saftParams);
    const auto [rfSaft, rfSaftCF] =
        uspam::beamformer::apply_saft<T, T>(timeDelay, rf);

    rfBeamformed = rfSaftCF;
  } break;

  case BeamformerType::NONE:
  default:
    rfBeamformed = rf; // Copy
  }
}

} // namespace uspam::beamformer
