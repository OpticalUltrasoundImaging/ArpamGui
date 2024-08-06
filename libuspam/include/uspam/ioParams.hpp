#pragma once

#include <armadillo>
#include <filesystem>
#include <opencv2/core.hpp>
#include <rapidjson/document.h>

namespace uspam::io {
namespace fs = std::filesystem;

constexpr int NUM_ALINES_DETAULT = 1000;
constexpr int RF_ALINE_SIZE = 8192;

// Container that holds coregistered PA and US data
template <typename T> struct PAUSpair {
  arma::Mat<T> PA;
  arma::Mat<T> US;

  template <typename Tother>
  static auto zeros_like(const PAUSpair<Tother> &other) {
    return PAUSpair{arma::Mat<T>(other.PA.n_rows, other.PA.n_cols),
                    arma::Mat<T>(other.US.n_rows, other.US.n_cols)};
  }

  template <typename Tother>
  static auto empty_like(const PAUSpair<Tother> &other) {
    return PAUSpair{
        arma::Mat<T>(other.PA.n_rows, other.PA.n_cols, arma::fill::none),
        arma::Mat<T>(other.US.n_rows, other.US.n_cols, arma::fill::none)};
  }
};

struct IOParams {
  int rf_size_PA{};
  int rf_size_spacer{};

  int offsetUS{};
  // Number of points
  int offsetPA{};

  // Byte offset at beginning of file.
  int byte_offset = 0;

public:
  [[nodiscard]] auto rf_size_US() const { return rf_size_PA * 2; }

  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline IOParams system2024v1() {
    // NOLINTNEXTLINE(*-magic-numbers)
    return IOParams{2650, 87, -100, -200, 1};
  }

  // System parameters from mid 2024 (ArpamGui acquisition)
  static inline IOParams system2024v2GUI() {
    // NOLINTNEXTLINE(*-magic-numbers)
    return IOParams{2650, 87, -100, -200, 1};
  }

  // Serialize to JSON
  [[nodiscard]] rapidjson::Document serializeToDoc() const;
  // NOLINTNEXTLINE(*-nodiscard)
  bool serializeToFile(const fs::path &path) const;

  // Deserialize from JSON
  bool deserialize(const rapidjson::Document &doc);
  bool deserializeFromFile(const fs::path &path);

  template <typename T>
  PAUSpair<T>
  allocateSplitPair(int alines_per_bscan = NUM_ALINES_DETAULT) const {
    arma::Mat<T> rfPA(this->rf_size_PA, alines_per_bscan, arma::fill::none);
    arma::Mat<T> rfUS(this->rf_size_US(), alines_per_bscan, arma::fill::none);
    return {rfPA, rfUS};
  }

  template <typename T1, typename T2>
  auto splitRfPAUS(const arma::Mat<T1> &rf, PAUSpair<T2> &split) const {
    const auto USstart = this->rf_size_PA + this->rf_size_spacer;
    const auto USend = USstart + this->rf_size_US();
    const auto offsetPA = this->offsetUS / 2 + this->offsetPA;

    assert(split.PA.size() == this->rf_size_PA * rf.n_cols);
    assert(split.US.size() == (USend - USstart) * rf.n_cols);

    for (int j = 0; j < rf.n_cols; ++j) {
      for (int i = 0; i < this->rf_size_PA; ++i) {
        split.PA(i, j) = static_cast<T2>(rf(i, j));
      }
      // US
      for (int i = 0; i < this->rf_size_US(); ++i) {
        split.US(i, j) = static_cast<T2>(rf(i + USstart, j));
      }

      {
        auto ptr = split.PA.colptr(j);
        std::rotate(ptr, ptr + offsetPA, ptr + split.PA.n_rows);
        // rfPA.rows(0, this->offset_PA - 1).zeros();
      }
      {
        auto ptr = split.US.colptr(j);
        std::rotate(ptr, ptr + this->offsetUS, ptr + split.US.n_rows);
        // rfUS.rows(0, this->offset_US - 1).zeros();
      }
    }
  }

  template <typename T1, typename Tb, typename Tout>
  auto splitRfPAUS_sub(const arma::Mat<T1> &rf, const arma::Col<Tb> &background,
                       arma::Mat<Tout> &rfPA, arma::Mat<Tout> &rfUS) const {

    const auto USstart = this->rf_size_PA + this->rf_size_spacer;
    const auto USend = USstart + this->rf_size_US();
    auto offsetUS = this->offsetUS;
    while (offsetUS < 0) {
      offsetUS = this->rf_size_US() + offsetUS;
    }
    auto offsetPA = this->offsetUS / 2 + this->offsetPA;
    while (offsetPA < 0) {
      offsetPA = this->rf_size_PA + offsetPA;
    }

    // Ensure rfPA and rfUS have enough space
    if (rfPA.n_rows != this->rf_size_PA || rfPA.n_cols != rf.n_cols) {
      rfPA.set_size(this->rf_size_PA, rf.n_cols);
    }
    if (rfUS.n_rows != this->rf_size_US() || rfUS.n_cols != rf.n_cols) {
      rfUS.set_size(this->rf_size_US(), rf.n_cols);
    }

    // Split
    cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
      for (int j = range.start; j < range.end; ++j) {

        // PA
        for (int i = 0; i < this->rf_size_PA; ++i) {
          // split.PA(i, j) = static_cast<Tout>(rf(i, j));
          rfPA(i, j) =
              static_cast<Tout>(static_cast<Tb>(rf(i, j)) - background(i));
        }
        // US
        for (int i = 0; i < this->rf_size_US(); ++i) {
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
               int nscans = 1, int alines_per_bscan = NUM_ALINES_DETAULT) const
      -> bool {
    // Only support nscans=1 for now
    assert(nscans >= 1);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to open file\n";
      return {};
    }

    constexpr auto byte_width = static_cast<int>(sizeof(T));
    const auto scan_size =
        RF_ALINE_SIZE * alines_per_bscan * nscans * byte_width;
    const auto start_pos = this->byte_offset + scan_size * i;

    file.seekg(start_pos, std::ios::beg);

    // Read file
    const auto matrix_size = RF_ALINE_SIZE * alines_per_bscan * byte_width;

    // NOLINTNEXTLINE(*-reinterpret-cast)
    if (!file.read(reinterpret_cast<char *>(storage.memptr()), matrix_size)) {
      std::cerr << "Failed to read file\n";
      return false;
    }
    return true;
  }

  template <typename T>
  [[nodiscard]] int
  get_num_scans(const fs::path &filename,
                int alines_per_bscan = NUM_ALINES_DETAULT) const {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "Failed to open file\n";
      return {};
    }
    file.seekg(0, std::ios::end);
    const std::streamsize fsize = file.tellg();
    constexpr auto byte_width = static_cast<int>(sizeof(T));
    const int numScans = (static_cast<int>(fsize) - this->byte_offset) /
                         (RF_ALINE_SIZE * alines_per_bscan * byte_width);
    return numScans;
  }
};
} // namespace uspam::io