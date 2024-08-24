#pragma once

#include <armadillo>
#include <filesystem>
#include <opencv2/core.hpp>
#include <rapidjson/document.h>

namespace uspam::io {
namespace fs = std::filesystem;

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

// Params related to reading the US or PA rf from the combined PAUS rf
struct IOParams_ {
  int start{}; // Where to start reading in the combined rf array
  int delay{}; // How much delay the start point is from the axis
  int size{};  // Num points to read from start
};

struct IOParams {

  int alinesPerBscan{};
  int samplesPerAscan{};

  IOParams_ PA;
  IOParams_ US;

  // Byte offset at beginning of file.
  int byteOffset = 0;

  // System parameters from early 2024 (Sitai Labview acquisition)
  static inline IOParams system2024v1() {
    // NOLINTBEGIN(*-magic-numbers)
    return IOParams{.alinesPerBscan = 1000,
                    .samplesPerAscan = 8192,
                    .PA =
                        {
                            .start = 0,
                            .delay = 150,
                            .size = 2580,
                        },
                    .US =
                        {
                            .start = 2800,
                            .delay = 0,
                            .size = 5460,
                        },
                    .byteOffset = 1};
    // NOLINTEND(*-magic-numbers)
  }

  // // System parameters from mid 2024 (ArpamGui acquisition)
  static inline IOParams system2024v2() {
    auto params = system2024v1();
    params.byteOffset = 0;

    params.PA.start = 0;
    params.PA.delay = 0;
    params.PA.size = 2800;
    params.US.start = 2800;
    params.US.delay = 0;
    params.US.size = 5460;
    return params;
  }

  /*
  Increasing DAQ buffer size from 8192 to 9600 to record deeper signal
  but truncate the first 500 PA points and first 1000 US points
  */
  static inline IOParams system2024v3() {
    // NOLINTBEGIN(*-magic-numbers)
    auto params = system2024v1();
    params.byteOffset = 0;

    params.PA.start = 0;
    params.PA.delay = 500;
    params.PA.size = 3200 - params.PA.delay;

    params.US.start = 2732;
    params.US.delay = 1000;
    params.US.size = 6400 - params.US.delay;

    // NOLINTEND(*-magic-numbers)
    return params;
  }

  /*
   */
  static inline IOParams convertedOldBin() {
    // NOLINTBEGIN(*-magic-numbers)
    auto params = system2024v1();
    params.byteOffset = 0;

    params.PA.start = 0;
    params.PA.delay = 500;
    params.PA.size = 3200 - params.PA.delay;

    params.US.start = 2732;
    params.US.delay = 1000;
    params.US.size = 6400 - params.US.delay;

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

  template <typename T> PAUSpair<T> allocateSplitPair() const {
    return {arma::Mat<T>(PA.size, alinesPerBscan, arma::fill::none),
            arma::Mat<T>(US.size, alinesPerBscan, arma::fill::none)};
  }

  template <typename T1, typename T2>
  void splitRfPAUS(const arma::Mat<T1> &rf, arma::Mat<T2> &rfPA,
                   arma::Mat<T2> &rfUS) const {
    rfPA.resize(PA.size, rf.n_cols);
    rfUS.resize(US.size, rf.n_cols);

    const auto range = cv::Range(0, rf.n_cols);
    // cv::parallel_for_(range, [&](const cv::Range &range) {
    for (int j = range.start; j < range.end; ++j) {
      const auto *pRF = rf.colptr(j);
      auto *pPA = rfPA.colptr(j);
      auto *pUS = rfUS.colptr(j);

      // NOLINTBEGIN(*-pointer-arithmetic)
      for (int i = 0; i < std::min(PA.size, US.start - PA.start); ++i) {
        pPA[i] = static_cast<T2>(pRF[i + PA.start]);
      }

      for (int i = 0; i < std::min(US.size, samplesPerAscan - US.start); ++i) {
        pUS[i] = static_cast<T2>(pRF[i + US.start]);
      }
      // NOLINTEND(*-pointer-arithmetic)
    }
    // });
  }

  // template <typename T1, typename Tb, typename Tout>
  // auto splitRfPAUS_sub(const arma::Mat<T1> &rf, const arma::Col<Tb>
  // &background,
  //                      arma::Mat<Tout> &PA, arma::Mat<Tout> &US) const {

  //   PA.resize(sizePA, rf.n_cols);
  //   US.resize(sizeUS(), rf.n_cols);

  //   cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
  //     for (int j = range.start; j < range.end; ++j) {
  //       const auto *pRF = rf.colptr(j);
  //       auto *pPA = PA.colptr(j);
  //       auto *pUS = US.colptr(j);

  //       // NOLINTBEGIN(*-pointer-arithmetic)
  //       for (int i = 0; i < sizePAacquired(); ++i) {
  //         pPA[i + sizePAdelay] =
  //             static_cast<Tout>(static_cast<Tb>(pRF[i]) - background[i]);
  //       }

  //       for (int i = 0; i < sizeUS(); ++i) {
  //         pUS[i] =
  //             static_cast<Tout>(static_cast<Tb>(pRF[i + sizePAacquired()]) -
  //                               background[i + sizePAacquired()]);
  //       }
  //       // NOLINTEND(*-pointer-arithmetic)

  //       {
  //         int middle = offsetUS;
  //         if (middle < 0) {
  //           middle = US.n_rows + middle;
  //         }
  //         std::rotate(pUS, pUS + middle, pUS + US.n_rows);
  //       }
  //     }
  //   });
  // };

  // // Split a single Aline
  // template <typename T> auto splitRfPAUS_aline(const arma::Col<T> &rf) const
  // {
  //   auto pair = allocateSplitPair<T>(1);
  //   splitRfPAUS(rf, pair);
  //   return pair;
  // }

  // // Split a Bscan
  // template <typename T> auto splitRfPAUS(const arma::Mat<T> &rf) const {
  //   auto pair = allocateSplitPair<T>();
  //   splitRfPAUS(rf, pair);
  //   return pair;
  // }

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