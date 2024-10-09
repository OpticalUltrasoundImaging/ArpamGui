#pragma once

#include <type_traits>
#include <uspam/beamformer/BeamformerParams.hpp>
#include <uspam/beamformer/BeamformerType.hpp>
#include <uspam/beamformer/SAFT.hpp>

namespace uspam::beamformer {

template <typename T>
void beamform(const arma::Mat<T> &rf, arma::Mat<T> &rfBeamformed,
              BeamformerType beamformer, BeamformerParams<T> beamformerParams,
              size_t truncate = 0)
  requires std::is_floating_point_v<T>
{
  switch (beamformer) {
  case BeamformerType::SAFT: {
    const auto timeDelay =
        uspam::beamformer::computeSaftTimeDelay<T>(beamformerParams);

    rfBeamformed = uspam::beamformer::apply_saft_v2<T, BeamformerType::SAFT>(
        timeDelay, rf, truncate);

  } break;

  case BeamformerType::SAFT_CF: {
    const auto timeDelay =
        uspam::beamformer::computeSaftTimeDelay<T>(beamformerParams);

    rfBeamformed = uspam::beamformer::apply_saft_v2<T, BeamformerType::SAFT_CF>(
        timeDelay, rf, truncate);
  } break;

  case BeamformerType::NONE:
  default:
    // Use "copy on write"
    // NOLINTNEXTLINE(*-const-cast)
    rfBeamformed = arma::Mat<T>(const_cast<T *>(rf.memptr()), rf.n_rows,
                                rf.n_cols, true, true);
  }
}

} // namespace uspam::beamformer
