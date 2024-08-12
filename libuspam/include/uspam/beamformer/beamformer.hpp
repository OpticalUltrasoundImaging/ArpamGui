#pragma once

#include <type_traits>
#include <uspam/beamformer/BeamformerType.hpp>
#include <uspam/beamformer/SAFT.hpp>
#include <variant>

namespace uspam::beamformer {

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
    rfBeamformed =
        uspam::beamformer::apply_saft<T, BeamformerType::SAFT>(timeDelay, rf);

  } break;

  case BeamformerType::SAFT_CF: {
    const auto saftParams =
        std::holds_alternative<SaftDelayParams<T>>(beamformerParams)
            ? std::get<SaftDelayParams<T>>(beamformerParams)
            : SaftDelayParams<T>::make();

    const auto timeDelay =
        uspam::beamformer::computeSaftTimeDelay<T>(saftParams);

    rfBeamformed = uspam::beamformer::apply_saft_v2<T, BeamformerType::SAFT_CF>(
        timeDelay, rf);
  } break;

  case BeamformerType::NONE:
  default:
    // Use "copy on write"
    // NOLINTNEXTLINE(*-const-cast)
    rfBeamformed = arma::Mat<T>(const_cast<T *>(rf.memptr()), rf.n_rows,
                                rf.n_cols, false, true);
  }
}

} // namespace uspam::beamformer
