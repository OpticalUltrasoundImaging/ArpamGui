#pragma once
#include "uspam/io.hpp"
#include <armadillo>
#include <filesystem>
#include <rapidjson/document.h>
#include <vector>

namespace uspam::recon {
namespace fs = std::filesystem;

struct ReconParams {
  std::vector<double> filterFreq;
  std::vector<double> filterGain;
  int noiseFloor;
  int desiredDynamicRange;
  int rotateOffset;

  void reconOneScan(arma::Mat<double> &rf, arma::Mat<uint8_t> &rfLog,
                    bool flip) const;
};

struct ReconParams2 {
  std::vector<double> filterFreqPA;
  std::vector<double> filterGainPA;
  std::vector<double> filterFreqUS;
  std::vector<double> filterGainUS;

  int noiseFloorPA;
  int noiseFloorUS;

  int desiredDynamicRangePA;
  int desiredDynamicRangeUS;

  int alineRotationOffset;

  static inline ReconParams2 system2024v1() {
    // NOLINTBEGIN(*-magic-numbers)
    return ReconParams2{{0, 0.03, 0.035, 0.2, 0.22, 1},
                        {0, 0, 1, 1, 0, 0},
                        {0, 0.1, 0.3, 1},
                        {0, 1, 1, 0},
                        300,
                        200,
                        35,
                        48,
                        26};
    // NOLINTEND(*-magic-numbers)
  }

  // Serialize to JSON
  [[nodiscard]] rapidjson::Document serializeToDoc() const;
  bool serializeToFile(const fs::path &path) const;

  // Deserialize from JSON
  bool deserialize(const rapidjson::Document &doc);
  bool deserializeFromFile(const fs::path &path);

  [[nodiscard]] inline ReconParams getPA() const {
    return ReconParams{filterFreqPA, filterGainPA, noiseFloorPA,
                       desiredDynamicRangePA, alineRotationOffset};
  }

  [[nodiscard]] inline ReconParams getUS() const {
    return ReconParams{filterFreqUS, filterGainUS, noiseFloorUS,
                       desiredDynamicRangeUS, alineRotationOffset};
  }
};

} // namespace uspam::recon