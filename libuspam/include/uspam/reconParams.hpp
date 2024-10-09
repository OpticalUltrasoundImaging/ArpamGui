#pragma once

#include "uspam/beamformer/SAFT.hpp"
#include "uspam/beamformer/beamformer.hpp"
#include <armadillo>
#include <filesystem>
#include <rapidjson/document.h>
#include <variant>

namespace uspam::recon {
namespace fs = std::filesystem;

using beamformer::BeamformerParams;
using beamformer::BeamformerType;

enum class FilterType { FIR, IIR };

struct ReconParams {
  /*
  System
  */
  float soundSpeed{1500.0F}; // [m/s] Speed of sound
  float fs{180.0e6F};        // [Hz] Sampling frequency

  /*
  Geometry
  */
  bool flipOnEven;
  int rotateOffset;

  /*
  Beamforming
  */
  BeamformerType beamformerType;
  BeamformerParams<float> beamformerParams;

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

  [[nodiscard]] auto flip(int frameIdx) const {
    const bool even = frameIdx % 2 == 0;
    return flipOnEven ? even : !even;
  }

  // [m]
  [[nodiscard]] auto pixelSpacing() const { return soundSpeed / fs; }
};

struct ReconParams2 {
  ReconParams PA;
  ReconParams US;

  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline ReconParams2 system2024v1() {
    // NOLINTBEGIN(*-magic-numbers)
    constexpr float soundSpeed = 1500.0F;
    constexpr float fs = 180.0e6F;
    constexpr int taps = 65;
    constexpr int order = 3;
    constexpr int rotateOffset = 25;
    constexpr bool flipOnEven = true;

    auto PAsaft = beamformer::SaftDelayParams<float>::make_PA();
    auto USsaft = beamformer::SaftDelayParams<float>::make_US();

    PAsaft.vs = soundSpeed;
    USsaft.vs = soundSpeed;

    ReconParams PA{.soundSpeed = soundSpeed,
                   .fs = fs,
                   .rotateOffset = rotateOffset,
                   .flipOnEven = flipOnEven,
                   .filterType = FilterType::FIR,
                   .firTaps = taps,
                   .iirOrder = order,
                   .bpLowFreq = 0.03,
                   .bpHighFreq = 0.22,
                   .truncate = 750,
                   .noiseFloor_mV = 9.0F,
                   .desiredDynamicRange = 30.0F,
                   .beamformerType = BeamformerType::SAFT_CF,
                   .beamformerParams = PAsaft};

    ReconParams US{.soundSpeed = soundSpeed,
                   .fs = fs,
                   .rotateOffset = rotateOffset,
                   .flipOnEven = flipOnEven,
                   .filterType = FilterType::IIR,
                   .firTaps = taps,
                   .iirOrder = order,
                   .bpLowFreq = 0.1,
                   .bpHighFreq = 0.3,
                   .truncate = 1000,
                   .noiseFloor_mV = 6.0F,
                   .desiredDynamicRange = 40.0F,
                   .beamformerType = BeamformerType::NONE,
                   .beamformerParams = USsaft};

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

  void updateSaftParamsFromReconParams() {
    if (std::holds_alternative<beamformer::SaftDelayParams<float>>(
            PA.beamformerParams)) {
      auto &bfParams =
          std::get<beamformer::SaftDelayParams<float>>(PA.beamformerParams);
      bfParams.vs = PA.soundSpeed;
    }

    if (std::holds_alternative<beamformer::SaftDelayParams<float>>(
            US.beamformerParams)) {
      auto &bfParams =
          std::get<beamformer::SaftDelayParams<float>>(US.beamformerParams);
      bfParams.vs = US.soundSpeed;
    }
  }
};

} // namespace uspam::recon
