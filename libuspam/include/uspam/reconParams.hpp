#pragma once

#include "uspam/SystemParams.hpp"
#include "uspam/beamformer/BeamformerType.hpp"
#include <armadillo>
#include <filesystem>
#include <rapidjson/document.h>

// NOLINTBEGIN(*-magic-numbers)
namespace uspam::recon {
namespace fs = std::filesystem;

using beamformer::BeamformerType;

enum class FilterType { FIR, IIR };

struct ReconParams {
  /*
  Beamforming
  */
  BeamformerType beamformerType;

  /*
  Filter
  */
  FilterType filterType;
  int firTaps;      // for FIR filter
  int iirOrder;     // for IIR filter
  float bpLowFreq;  // Bandpass low freq
  float bpHighFreq; // Bandpass high freq

  /*
  Log compression
  */
  float noiseFloor_mV;
  float desiredDynamicRange; // [dB]

  int truncate; // num samples at the beginning to zero (pulser/laser artifacts)

  /*
  Serialization
  */
  [[nodiscard]] rapidjson::Value
  serialize(rapidjson::Document::AllocatorType &allocator) const;
  static ReconParams deserialize(const rapidjson::Value &obj);
};

struct ReconParams2 {
  SystemParams system;
  ReconParams PA;
  ReconParams US;

  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline ReconParams2 system2024v1() {

    SystemParams system;
    system.soundSpeed = 1500.0F;
    system.fs = 180.0e6F;
    system.rotateOffset = 25;
    system.flipOnEven = true;

    constexpr int taps = 65;
    constexpr int order = 3;

    ReconParams PA{
        .beamformerType = BeamformerType::SAFT_CF,
        .filterType = FilterType::FIR,
        .firTaps = taps,
        .iirOrder = order,
        .bpLowFreq = 0.03,
        .bpHighFreq = 0.22,
        .noiseFloor_mV = 9.0F,
        .desiredDynamicRange = 30.0F,
        .truncate = 750,
    };

    ReconParams US{
        .beamformerType = BeamformerType::NONE,
        .filterType = FilterType::IIR,
        .firTaps = taps,
        .iirOrder = order,
        .bpLowFreq = 0.1,
        .bpHighFreq = 0.3,
        .noiseFloor_mV = 6.0F,
        .desiredDynamicRange = 40.0F,
        .truncate = 1000,
    };

    return ReconParams2{system, PA, US};
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline ReconParams2 system2024v2GUI() {
    auto params = system2024v1();
    params.system.rotateOffset = 0;
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

// NOLINTEND(*-magic-numbers)