#pragma once

#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/beamformer/SAFT.hpp"
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
  int medfiltKsize{0};

  FilterType filterType;
  int firTaps;  // for FIR filter
  int iirOrder; // for IIR filter

  float bpLowFreq;  // Bandpass low freq
  float bpHighFreq; // Bandpass high freq

  int padding;  // Padding due to distance between transducer surface and the
                // rotation axis
  int truncate; // num samples at the beginning to zero (pulser/laser artifacts)

  int rotateOffset;

  float noiseFloor_mV;
  float desiredDynamicRange;

  bool flipOnEven; // If true, flip on even. else flip on odd

  bool findSurface{true};
  bool showSurface{false};
  bool cleanSurface{false};
  int additionalSamplesToCleanSurface{0};

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

    constexpr int padding = 250;
    constexpr int rotateOffset = 25;
    constexpr bool flipOnEven = true;

    ReconParams PA{.medfiltKsize = 3,
                   .filterType = FilterType::FIR,
                   .firTaps = taps,
                   .iirOrder = order,
                   .bpLowFreq = 0.03,
                   .bpHighFreq = 0.22,
                   .padding = padding,
                   .truncate = 250,
                   .rotateOffset = rotateOffset,
                   .noiseFloor_mV = 6.0F,
                   .desiredDynamicRange = 30.0F,
                   .flipOnEven = flipOnEven,
                   .beamformerType = BeamformerType::SAFT_CF,
                   .beamformerParams =
                       beamformer::SaftDelayParams<float>::make_PA()};
    ReconParams US{.medfiltKsize = 3,
                   .filterType = FilterType::IIR,
                   .firTaps = taps,
                   .iirOrder = order,
                   .bpLowFreq = 0.1,
                   .bpHighFreq = 0.3,
                   .padding = padding * 2,
                   .truncate = 500,
                   .rotateOffset = rotateOffset,
                   .noiseFloor_mV = 3.0F,
                   .desiredDynamicRange = 40.0F,
                   .flipOnEven = flipOnEven,
                   .beamformerType = BeamformerType::NONE,
                   .beamformerParams =
                       beamformer::SaftDelayParams<float>::make_US()};

    return ReconParams2{PA, US};
    // NOLINTEND(*-magic-numbers)
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline ReconParams2 system2024v2probe1() {
    auto params = system2024v1();
    // NOLINTBEGIN(*-magic-numbers)
    params.PA.rotateOffset = 0;
    params.US.rotateOffset = 0;
    // NOLINTEND(*-magic-numbers)
    return params;
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline ReconParams2 system2024v2probe2() {
    auto params = system2024v2probe1();
    // NOLINTBEGIN(*-magic-numbers)
    params.PA.noiseFloor_mV = 6.0;
    params.US.noiseFloor_mV = 3.0;
    params.US.desiredDynamicRange = 50;

    params.PA.flipOnEven = false;
    params.US.flipOnEven = false;
    // NOLINTEND(*-magic-numbers)
    return params;
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  // Deeper acquisition
  static inline ReconParams2 system2024v3() {
    auto params = system2024v1();
    // NOLINTBEGIN(*-magic-numbers)
    params.PA.rotateOffset = 0;
    params.US.rotateOffset = 0;

    params.PA.medfiltKsize = 5;
    params.PA.beamformerType = BeamformerType::NONE;

    params.PA.truncate = 50;
    params.US.truncate = 100;
    // NOLINTEND(*-magic-numbers)
    return params;
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline ReconParams2 convertedOldBin() {
    auto params = system2024v1();
    // NOLINTBEGIN(*-magic-numbers)
    params.PA.rotateOffset = 0;
    params.US.rotateOffset = 0;

    params.PA.medfiltKsize = 5;
    params.PA.beamformerType = BeamformerType::NONE;

    params.US.medfiltKsize = 3;
    params.US.noiseFloor_mV = 4.0;

    params.PA.truncate = 0;
    params.US.truncate = 0;

    params.PA.flipOnEven = false;
    params.US.flipOnEven = false;

    params.PA.cleanSurface = true;
    params.PA.additionalSamplesToCleanSurface = 90;
    params.US.cleanSurface = true;

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
