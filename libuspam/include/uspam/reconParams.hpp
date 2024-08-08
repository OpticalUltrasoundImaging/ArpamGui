#pragma once

#include "uspam/beamformer/beamformer.hpp"
#include <armadillo>
#include <filesystem>
#include <rapidjson/document.h>
#include <tuple>
#include <vector>

namespace uspam::recon {
namespace fs = std::filesystem;

using beamformer::BeamformerParams;
using beamformer::BeamformerType;

struct ReconParams {
  int butterOrder;
  float bpLowFreq;  // Bandpass low freq
  float bpHighFreq; // Bandpass high freq

  int truncate; // num samples at the beginning to zero (pulser/laser artifacts)
  int rotateOffset;
  float noiseFloor_mV;
  float desiredDynamicRange;

  BeamformerType beamformerType;
  BeamformerParams<float> beamformerParams;

  [[nodiscard]] rapidjson::Value
  serialize(rapidjson::Document::AllocatorType &allocator) const;
  static ReconParams deserialize(const rapidjson::Value &obj);

  static bool flip(int frameIdx) { return frameIdx % 2 == 0; }
};

struct ReconParams2 {
  ReconParams PA;
  ReconParams US;

  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline ReconParams2 system2024v1() {
    // NOLINTBEGIN(*-magic-numbers)
    constexpr int rotateOffset = 25;
    ReconParams PA{
        3, 0.03, 0.22, 500, rotateOffset, 7.0F, 30.0F, BeamformerType::SAFT_CF};
    ReconParams US{
        3, 0.1, 0.3, 500, rotateOffset, 6.0F, 40.0F, BeamformerType::NONE};

    return ReconParams2{PA, US};
    // NOLINTEND(*-magic-numbers)
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline ReconParams2 system2024v2GUI() {
    // NOLINTBEGIN(*-magic-numbers)
    ReconParams PA{3, 0.03, 0.22, 500, 0, 7.0F, 30.0F, BeamformerType::SAFT_CF};
    ReconParams US{3, 0.1, 0.3, 500, 0, 6.0F, 40.0F, BeamformerType::NONE};

    return ReconParams2{PA, US};
    // NOLINTEND(*-magic-numbers)
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
