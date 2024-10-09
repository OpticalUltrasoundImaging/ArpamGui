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

  int offsetUS{}; // [samples]
  int offsetPA{}; // [samples]

  // Byte offset at beginning of file.
  int byteOffset = 0;

public:
  [[nodiscard]] auto rfSizeUS() const { return rfSizePA * 2; }

  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline IOParams system2024v1() {
    return IOParams{.alinesPerBscan = 1000,
                    .rfSizePA = 2650,
                    .offsetUS = -100,
                    .offsetPA = -200,
                    .byteOffset = 1};
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline IOParams system2024v2GUI() {
    auto params = system2024v1();
    params.byteOffset = 0;
    params.offsetPA = -100;
    return params;
  }

  // Serialize to JSON
  [[nodiscard]] rapidjson::Document serializeToDoc() const;
  // NOLINTNEXTLINE(*-nodiscard)
  bool serializeToFile(const fs::path &path) const;

  // Deserialize from JSON
  bool deserialize(const rapidjson::Document &doc);
  bool deserializeFromFile(const fs::path &path);

  template <typename T1, typename T2>
  void splitRfPAUS(const arma::Mat<T1> &rf, arma::Mat<T2> &PA,
                   arma::Mat<T2> &US) const {
    const auto USstart = this->rfSizePA;

    PA.resize(rfSizePA, rf.n_cols);
    US.resize(rfSizeUS(), rf.n_cols);

    cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
      for (int j = range.start; j < range.end; ++j) {
        const auto *rfcol = rf.colptr(j);
        auto *PAcol = PA.colptr(j);
        auto *UScol = US.colptr(j);

        // NOLINTBEGIN(*-pointer-arithmetic)
        for (int i = 0; i < rfSizePA; ++i) {
          PAcol[i] = static_cast<T2>(rfcol[i]);
        }

        for (int i = 0; i < rfSizeUS(); ++i) {
          UScol[i] = static_cast<T2>(rfcol[i + USstart]);
        }
        // NOLINTEND(*-pointer-arithmetic)

        {
          int middle = offsetPA;
          if (middle < 0) {
            middle = PA.n_rows + middle;
          }
          std::rotate(PAcol, PAcol + middle, PAcol + PA.n_rows);
        }

        {
          int middle = offsetUS;
          if (middle < 0) {
            middle = US.n_rows + middle;
          }
          std::rotate(UScol, UScol + middle, UScol + US.n_rows);
        }
      }
    });
  }

  template <typename T1, typename Tb, typename Tout>
  auto splitRfPAUS_sub(const arma::Mat<T1> &rf, const arma::Col<Tb> &background,
                       arma::Mat<Tout> &rfPA, arma::Mat<Tout> &rfUS) const {

    const auto USstart = this->rfSizePA;
    auto offsetUS = this->offsetUS;
    while (offsetUS < 0) {
      offsetUS = this->rfSizeUS() + offsetUS;
    }
    auto offsetPA = this->offsetPA;
    while (offsetPA < 0) {
      offsetPA = this->rfSizePA + offsetPA;
    }

    // Ensure rfPA and rfUS have enough space
    rfPA.set_size(rfSizePA, rf.n_cols);
    rfUS.set_size(rfSizeUS(), rf.n_cols);

    // Split
    cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
      for (int j = range.start; j < range.end; ++j) {

        // PA
        for (int i = 0; i < this->rfSizePA; ++i) {
          // split.PA(i, j) = static_cast<Tout>(rf(i, j));
          rfPA(i, j) =
              static_cast<Tout>(static_cast<Tb>(rf(i, j)) - background(i));
        }

        // US
        for (int i = 0; i < this->rfSizeUS(); ++i) {
          rfUS(i, j) = static_cast<Tout>(static_cast<Tb>(rf(i + USstart, j)) -
                                         background(i + USstart));
        }

        {
          auto ptr = rfPA.colptr(j);
          std::rotate(ptr, ptr + offsetPA, ptr + rfPA.n_rows);
          // rfPA.rows(0, this->offset_PA - 1).zeros();
        }

        {
          auto ptr = rfUS.colptr(j);
          std::rotate(ptr, ptr + offsetUS, ptr + rfUS.n_rows);
          // rfUS.rows(0, this->offset_US - 1).zeros();
        }
      }
    });
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
