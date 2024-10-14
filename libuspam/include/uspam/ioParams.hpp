#pragma once

#include <armadillo>
#include <filesystem>
#include <opencv2/core.hpp>
#include <rapidjson/document.h>

// NOLINTBEGIN(*-magic-numbers)

namespace uspam::io {
namespace fs = std::filesystem;

constexpr int RF_ALINE_SIZE = 8192;

struct IOParams {
  int alinesPerBscan{};

  int rfSizePA{};
  [[nodiscard]] auto rfSizeUS() const { return rfSizePA * 2; }

  /*
  By default, PA starts at 0 and US starts at rfSizePA.
  This can be adjusted with offsetPA and offsetUS,
  PA will start at offsetPA and US will start at offsetUS

  For example, if offsetPA = -100 (signal starts 100 points before the trigger
  was received), then rfPA will contain 100 empty points at the beginning, while
  if offsetPA is 100, then the first rfPA will be the 100th element of rf.

  */

  int offsetPA{}; // [samples]
  int offsetUS{}; // [samples]

  // Byte offset at beginning of file.
  int byteOffset = 0;

public:
  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline IOParams system2024v1() {
    return IOParams{.alinesPerBscan = 1000,
                    .rfSizePA = 2730,
                    .offsetPA = -200,
                    .offsetUS = 0,
                    .byteOffset = 1};
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline IOParams system2024v2GUI() {
    auto params = system2024v1();
    params.byteOffset = 0;
    params.rfSizePA = 2750;
    params.offsetPA = 0;
    params.offsetUS = -20;
    return params;
  }

  // Serialize to JSON
  [[nodiscard]] rapidjson::Document serializeToDoc() const;
  // NOLINTNEXTLINE(*-nodiscard)
  bool serializeToFile(const fs::path &path) const;

  // Deserialize from JSON
  bool deserialize(const rapidjson::Document &doc);
  bool deserializeFromFile(const fs::path &path);

  template <typename T1, typename Tb, typename Tout>
  void splitRfPAUS(const arma::Mat<T1> &rf, arma::Mat<Tout> &rfPA,
                   arma::Mat<Tout> &rfUS, const arma::Col<Tb> &background = {},
                   const bool subtractPA = true,
                   const bool subtractUS = true) const {
    // If subtractPA or subtractUS is true, background must not be empty
    assert(!subtractPA || !background.empty());
    assert(!subtractUS || !background.empty());

    // PA start and end indices in rf
    const auto PAstart = std::max(0, this->offsetPA);
    const auto PAend =
        std::min(this->rfSizePA, this->rfSizePA - std::abs(this->offsetPA));

    // US start and end indices in rf
    const auto USstart = std::max<int>(
        0, std::max(this->rfSizePA, this->rfSizePA + this->offsetUS));
    const auto USend =
        std::min<int>(rf.n_rows, std::min(this->rfSizePA + this->rfSizeUS(),
                                          this->rfSizePA + this->rfSizeUS() -
                                              std::abs(this->offsetUS)));

    // PA and US start in rfPA and rfUS
    const auto rfPAstart = std::max(0, -this->offsetPA);
    const auto rfUSstart = std::max(0, -this->offsetUS);

    // Ensure rfPA and rfUS have enough space
    rfPA.set_size(rfSizePA, rf.n_cols);
    rfUS.set_size(rfSizeUS(), rf.n_cols);

    // Split
    const auto range = cv::Range(0, rf.n_cols);
    // cv::parallel_for_(range, [&](const cv::Range &range) {
    for (int j = range.start; j < range.end; ++j) {

      // PA
      int rfPAidx = rfPAstart;
      for (int i = PAstart; i < PAend; ++i) {
        if (subtractPA) {
          rfPA(rfPAidx++, j) = (Tout)(rf(i, j) - background(i));
        } else {
          rfPA(rfPAidx++, j) = (Tout)rf(i, j);
        }
      }

      // US
      int rfUSidx = rfUSstart;
      for (int i = USstart; i < USend; ++i) {
        if (subtractUS) {
          rfUS(rfUSidx++, j) = (Tout)(rf(i, j) - background(i));
        } else {
          rfUS(rfUSidx++, j) = (Tout)rf(i, j);
        }
      }
    }
    // });
  };

  // Split a single Aline
  template <typename T> auto splitRfPAUS_aline(const arma::Col<T> &rf) const {
    auto pair = allocateSplitPair<T>(1);
    splitRfPAUS(rf, pair);
    return pair;
  }

  // Split a Bscan
  template <typename T> auto splitRfPAUS(const arma::Mat<T> &rf) const {
    auto pair = allocateSplitPair<T>();
    splitRfPAUS(rf, pair);
    return pair;
  }

  // template <typename T>
  // auto load_rf(const fs::path &filename, int i, int nscans = 1,
  //              int alines_per_bscan = 1000) const -> arma::Mat<T>;
  template <typename T>
  auto load_rf(const fs::path &filename, arma::Mat<T> &storage, int i,
               int nscans = 1) const -> bool {
    // Only support nscans=1 for now
    assert(nscans >= 1);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to open file\n";
      return {};
    }

    constexpr auto byte_width = static_cast<int>(sizeof(T));
    const auto scan_size = RF_ALINE_SIZE * alinesPerBscan * nscans * byte_width;
    const auto start_pos = this->byteOffset + scan_size * i;

    file.seekg(start_pos, std::ios::beg);

    // Read file
    const auto matrix_size = RF_ALINE_SIZE * alinesPerBscan * byte_width;

    // NOLINTNEXTLINE(*-reinterpret-cast)
    if (!file.read(reinterpret_cast<char *>(storage.memptr()), matrix_size)) {
      std::cerr << "Failed to read file\n";
      return false;
    }
    return true;
  }

  template <typename T>
  [[nodiscard]] int get_num_scans(const fs::path &filename) const {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "Failed to open file\n";
      return {};
    }
    file.seekg(0, std::ios::end);
    const std::streamsize fsize = file.tellg();
    constexpr auto byte_width = static_cast<int>(sizeof(T));
    const int numScans = (static_cast<int>(fsize) - this->byteOffset) /
                         (RF_ALINE_SIZE * alinesPerBscan * byte_width);
    return numScans;
  }
};
} // namespace uspam::io

// NOLINTEND(*-magic-numbers)
