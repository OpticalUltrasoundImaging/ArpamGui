#pragma once

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace uspam::beamformer {

enum class BeamformerType {
  NONE,
  SAFT,
  SAFT_CF,
};

namespace details {

using PairT = std::pair<BeamformerType, std::string_view>;
static constexpr std::array BFTypeToString{
    PairT{BeamformerType::NONE, "NONE"},
    PairT{BeamformerType::SAFT, "SAFT"},
    PairT{BeamformerType::SAFT_CF, "SAFT_CF"},
};

} // namespace details

inline constexpr auto BeamformerTypeToString(const BeamformerType btype) {
  using details::BFTypeToString;
  const auto it =
      std::find_if(BFTypeToString.cbegin(), BFTypeToString.cend(),
                   [=](const auto &pair) { return pair.first == btype; });
  return (it != BFTypeToString.cend()) ? it->second : "";
}

inline constexpr auto BeamformerTypeFromString(const std::string_view str) {
  {
    using details::BFTypeToString;
    const auto it =
        std::find_if(BFTypeToString.cbegin(), BFTypeToString.cend(),
                     [=](const auto &pair) { return pair.second == str; });
    return it != BFTypeToString.cend() ? it->first : BeamformerType::NONE;
  }
}

} // namespace uspam::beamformer