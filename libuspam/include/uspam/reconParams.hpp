#pragma once

#include "uspam/beamformer/beamformer.hpp"
#include <armadillo>
#include <filesystem>
#include <rapidjson/document.h>

namespace uspam::recon {
namespace fs = std::filesystem;

using beamformer::BeamformerParams;
using beamformer::BeamformerType;

enum class FilterType { FIR, IIR };

struct ReconParams {
  FilterType filterType;

  int firTaps;  // for FIR filter
  int iirOrder; // for IIR filter

  float bpLowFreq;  // Bandpass low freq
  float bpHighFreq; // Bandpass high freq

  int truncate; // num samples at the beginning to zero (pulser/laser artifacts)
  int rotateOffset;
  float noiseFloor_mV;
  float desiredDynamicRange;

  bool flipOnEven; // If true, flip on even. else flip on odd

  BeamformerType beamformerType;
  BeamformerParams<float> beamformerParams;

  [[nodiscard]] rapidjson::Value
  serialize(rapidjson::Document::AllocatorType &allocator) const;
  static ReconParams deserialize(const rapidjson::Value &obj);

  [[nodiscard]] auto flip(int frameIdx) const {
    const bool even = frameIdx % 2 == 0;
    return flipOnEven ? even : !even;
  }
};

struct ReconParams2 {
  ReconParams PA;
  ReconParams US;

  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline ReconParams2 system2024v1() {
    // NOLINTBEGIN(*-magic-numbers)
    constexpr int taps = 95;
    constexpr int order = 3;
    constexpr int rotateOffset = 25;
    constexpr bool flipOnEven = true;
    ReconParams PA{FilterType::FIR,
                   taps,
                   order,
                   0.03,
                   0.22,
                   500,
                   rotateOffset,
                   9.0F,
                   30.0F,
                   flipOnEven,
                   BeamformerType::SAFT_CF};
    ReconParams US{FilterType::IIR,
                   taps,
                   order,
                   0.1,
                   0.3,
                   500,
                   rotateOffset,
                   6.0F,
                   40.0F,
                   flipOnEven,
                   BeamformerType::NONE};

    return ReconParams2{PA, US};
    // NOLINTEND(*-magic-numbers)
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline ReconParams2 system2024v2GUI() {
    auto params = system2024v1();
    // NOLINTBEGIN(*-magic-numbers)
    params.PA.rotateOffset = 0;
    params.US.rotateOffset = 0;
    // NOLINTEND(*-magic-numbers)
    return params;
  }

  // Serialize to JSON
  [[nodiscard]] rapidjson::Document serializeToDoc() const;
  // NOLINTNEXTLINE(*-nodiscard)
  bool serializeToFile(const fs::path &path) const;

  // Deserialize from JSON
  bool deserialize(const rapidjson::Document &doc);
  bool deserializeFromFile(const fs::path &path);
};

} // namespace uspam::recon
