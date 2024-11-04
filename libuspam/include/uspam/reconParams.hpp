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
  bool backgroundSubtract{false};

  /*
  Beamforming
  */
  BeamformerType beamformerType{BeamformerType::NONE};

  /*
  Denoising filter
  */
  int medFiltSize{1};

  /*
  Bandpass Filter
  */
  FilterType filterType{FilterType::FIR};
  int firTaps{65};       // for FIR filter
  int iirOrder{3};       // for IIR filter
  float bpLowFreq{0.1};  // Bandpass low freq
  float bpHighFreq{0.3}; // Bandpass high freq

  /*
  Log compression
  */
  float noiseFloor_mV{3.0F};
  float desiredDynamicRange{40.0}; // [dB]

  int truncate{200}; // num samples at the beginning to zero (pulser/laser
                     // artifacts)

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
        .backgroundSubtract = true,
        .beamformerType = BeamformerType::SAFT_CF,
        .medFiltSize = 3,
        .filterType = FilterType::FIR,
        .firTaps = taps,
        .iirOrder = order,
        .bpLowFreq = 0.03,
        .bpHighFreq = 0.22,
        .noiseFloor_mV = 5.0F,
        .desiredDynamicRange = 30.0F,
        .truncate = 200,
    };

    ReconParams US{
        .backgroundSubtract = false,
        .beamformerType = BeamformerType::NONE,
        .medFiltSize = 1,
        .filterType = FilterType::IIR,
        .firTaps = taps,
        .iirOrder = order,
        .bpLowFreq = 0.1,
        .bpHighFreq = 0.3,
        .noiseFloor_mV = 5.0F,
        .desiredDynamicRange = 30.0F,
        .truncate = 400,
    };

    return ReconParams2{system, PA, US};
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline ReconParams2 system2024v2GUI() {
    auto params = system2024v1();
    params.system.rotateOffset = 5;
    params.PA.truncate = 500;
    params.US.truncate = 1000;
    return params;
  }

  static inline ReconParams2 system2024v2GUIprobe2() {
    auto params = system2024v2GUI();
    params.system.flipOnEven = false;
    params.PA.noiseFloor_mV = 4.0;
    params.US.noiseFloor_mV = 3.0;
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