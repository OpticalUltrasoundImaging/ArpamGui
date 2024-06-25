#pragma once

#include <armadillo>
#include <filesystem>
#include <rapidjson/document.h>
#include <vector>

namespace uspam::recon {
namespace fs = std::filesystem;

struct ReconParams {
  std::vector<double> filterFreq;
  std::vector<double> filterGain;
  int truncate; // num samples at the beginning to zero (pulser/laser artifacts)
  int rotateOffset;
  float noiseFloor_mV;
  float desiredDynamicRange;

  [[nodiscard]] rapidjson::Value
  serialize(rapidjson::Document::AllocatorType &allocator) const;
  static ReconParams deserialize(const rapidjson::Value &obj);

  static bool flip(int frameIdx) { return frameIdx % 2 == 0; }
};

struct ReconParams2 {
  ReconParams PA;
  ReconParams US;

  static inline ReconParams2 system2024v1() {
    // NOLINTBEGIN(*-magic-numbers)
    ReconParams PA{{0, 0.03, 0.035, 0.2, 0.22, 1},
                   {0, 0, 1, 1, 0, 0},
                   250,
                   25,
                   9.0f,
                   35.0f};
    ReconParams US{{0, 0.1, 0.3, 1}, {0, 1, 1, 0}, 500, 25, 6.0f, 48.0f};

    return ReconParams2{PA, US};
    // NOLINTEND(*-magic-numbers)
  }

  // Serialize to JSON
  [[nodiscard]] rapidjson::Document serializeToDoc() const;
  bool serializeToFile(const fs::path &path) const;

  // Deserialize from JSON
  bool deserialize(const rapidjson::Document &doc);
  bool deserializeFromFile(const fs::path &path);
};

} // namespace uspam::recon
