#pragma once

#include "uspam/beamformer/SAFT.hpp"
#include <type_traits>
#include <variant>

namespace uspam::beamformer {

enum class BeamformerType {
  NONE,
  SAFT,
  SAFT_CF,
};

template <typename T>
using BeamformerParams = std::variant<std::monostate, SaftDelayParams<T>>;

template <typename T>
void beamform(const arma::Mat<T> &rf, arma::Mat<T> &rfBeamformed,
              BeamformerType beamformer, BeamformerParams<T> beamformerParams)
  requires std::is_floating_point_v<T>
{
  switch (beamformer) {
  case BeamformerType::SAFT: {
    const auto saftParams =
        std::holds_alternative<SaftDelayParams<T>>(beamformerParams)
            ? std::get<SaftDelayParams<T>>(beamformerParams)
            : SaftDelayParams<T>::make();

    const auto timeDelay =
        uspam::beamformer::computeSaftTimeDelay<T>(saftParams);
    const auto [rfSaft, rfSaftCF] =
        uspam::beamformer::apply_saft<T, T>(timeDelay, rf);

    rfBeamformed = rfSaft;
  } break;

  case BeamformerType::SAFT_CF: {

    const auto saftParams =
        std::holds_alternative<SaftDelayParams<T>>(beamformerParams)
            ? std::get<SaftDelayParams<T>>(beamformerParams)
            : SaftDelayParams<T>::make();

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
