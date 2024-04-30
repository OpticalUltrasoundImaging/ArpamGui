#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <span>
#include <string>

#include <armadillo>

namespace uspam::io {
namespace fs = std::filesystem;

const int RF_ALINE_SIZE = 8192;

// Swap the endianness of a value inplace
template <typename T> void swap_endian_inplace(T *val) {
  char *ptr = reinterpret_cast<char *>(val); // NOLINT
  std::reverse(ptr, ptr + sizeof(T));        // NOLINT
}

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
  // No. of samples in US RF. Must be twice rf_size_PA
  int rf_size_US{};

  int offsetUS{};
  // Number of points
  int offsetPA{};

  // Byte offset at beginning of file.
  int byte_offset = 0;

  // System parameters from early 2024
  static inline IOParams system2024v1() {
    return IOParams{2650, 87, 5300, 350, 215, 1};
  }

  template <typename T>
  PAUSpair<T> allocateSplitPair(int alines_per_bscan = 1000) const {
    arma::Mat<T> rfPA(this->rf_size_PA, alines_per_bscan, arma::fill::none);
    arma::Mat<T> rfUS(this->rf_size_US, alines_per_bscan, arma::fill::none);
    return {rfPA, rfUS};
  }

  template <typename T1, typename T2>
  auto splitRfPAUS(const arma::Mat<T1> &rf, PAUSpair<T2> &split) const {
    const auto USstart = this->rf_size_PA + this->rf_size_spacer;
    const auto USend = USstart + this->rf_size_US;
    const auto offsetPA = this->offsetUS / 2 + this->offsetPA;

    assert(split.PA.size() == this->rf_size_PA * rf.n_cols);
    assert(split.US.size() == (USend - USstart) * rf.n_cols);

    // split.PA = rf.rows(0, this->rf_size_PA - 1);
    // split.US = rf.rows(USstart, USend - 1);
    for (int j = 0; j < rf.n_cols; ++j) {
      // PA
      for (int i = 0; i < this->rf_size_PA; ++i) {
        split.PA(i, j) = static_cast<T2>(rf(i, j));
      }
      // US
      for (int i = 0; i < this->rf_size_US; ++i) {
        split.US(i, j) = static_cast<T2>(rf(i + USstart, j));
      }
    }

    for (int i = 0; i < rf.n_cols; ++i) {
      {
        auto ptr = split.PA.colptr(i);
        std::rotate(ptr, ptr + this->offsetPA, ptr + split.PA.n_rows);
        // rfPA.rows(0, this->offset_PA - 1).zeros();
      }
      {
        auto ptr = split.US.colptr(i);
        std::rotate(ptr, ptr + this->offsetUS, ptr + split.US.n_rows);
        // rfUS.rows(0, this->offset_US - 1).zeros();
      }
    }
  }

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
               int nscans = 1, int alines_per_bscan = 1000) const -> bool {
    // Only support nscans=1 for now
    assert(nscans >= 1);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to open file\n";
      return {};
    }

    const auto scan_size =
        RF_ALINE_SIZE * alines_per_bscan * nscans * sizeof(T);
    const auto start_pos = this->byte_offset + scan_size * i;

    file.seekg(start_pos, std::ios::beg);

    // Read file
    const auto matrix_size = RF_ALINE_SIZE * alines_per_bscan * sizeof(T);
    if (!file.read(reinterpret_cast<char *>(storage.memptr()), matrix_size)) {
      std::cerr << "Failed to read file\n";
      return false;
    }
    return true;
  }

  template <typename T>
  int get_num_scans(const fs::path &filename,
                    int alines_per_bscan = 1000) const {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "Failed to open file\n";
      return {};
    }
    file.seekg(0, std::ios::end);
    const std::streamsize fsize = file.tellg();
    int numScans = (fsize - this->byte_offset) /
                   (RF_ALINE_SIZE * alines_per_bscan * sizeof(T));
    return numScans;
  }
};

template <typename TypeInBin> class BinfileLoader {
private:
  std::ifstream file;
  int byteOffset = 0;
  int numScans = 0;
  int alinesPerBscan = 0;
  int currScanIdx = 0;
  std::mutex mtx;

public:
  BinfileLoader(const IOParams &ioparams, const fs::path filename,
                int alinesPerBscan = 1000)
      : file(filename, std::ios::binary | std::ios::ate),
        byteOffset(ioparams.byte_offset), alinesPerBscan(alinesPerBscan) {
    if (!file.is_open()) {
      throw std::runtime_error(
          std::string("[BinfileLoader] Failed to open file ") +
          filename.generic_string());
    }
    const std::streamsize fsize = file.tellg();
    numScans = (fsize - ioparams.byte_offset) / scanSizeBytes();
    file.seekg(ioparams.byte_offset, std::ios::beg);
  }

  // (bytes) Raw RF size of one PAUS scan
  auto scanSizeBytes() const {
    return RF_ALINE_SIZE * alinesPerBscan * sizeof(TypeInBin);
  }

  auto size() const { return numScans; }

  void setCurrIdx(int idx) {
    std::lock_guard lock(mtx);
    assert(idx >= 0 && idx < numScans);
    currScanIdx = idx;
  }

  bool hasMoreScans() {
    std::lock_guard lock(mtx);
    return currScanIdx < numScans;
  }

  bool get(arma::Mat<TypeInBin> &rf) {
    std::lock_guard lock(mtx);
    assert(currScanIdx < numScans);
    assert(rf.size() * sizeof(TypeInBin) == scanSizeBytes());

    const auto sizeBytes = scanSizeBytes();
    const auto start_pos = this->byteOffset + sizeBytes * currScanIdx;
    file.seekg(start_pos, std::ios::beg);

    // Read file
    return !file.read(reinterpret_cast<char *>(rf.memptr()), sizeBytes);
  }

  auto getNext(arma::Mat<TypeInBin> &rfStorage) {
    get(rfStorage);
    std::lock_guard lock(mtx);
    currScanIdx++;
  }
};

// T is the type of value stored in the binary file.
template <typename T>
auto load_bin(const fs::path &filename,
              const std::endian endian = std::endian::little) -> arma::Mat<T> {

  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Failed to open file\n";
    return {};
  }
  const std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  const auto n_values = fsize / sizeof(T);
  const size_t cols = 1000;
  const size_t rows = n_values / cols;

  // Check if the file size matches our matrix size
  if (rows * cols * sizeof(T) != fsize) {
    std::cerr << "File size does not match the expected matrix dimensions\n";
    return {};
  }

  arma::Mat<T> matrix(rows, cols);

  // Read file
  if (!file.read(
          reinterpret_cast<char *>(matrix.data()), // NOLINT(*-reinterpret-cast)
          fsize)) {
    std::cerr << "Failed to read data into matrix\n";
    return {};
  }

  // Convert endian if necessary
  if (endian != std::endian::native) {
    const auto ptr = matrix.data();
    for (int i = 0; i < matrix.size(); i++) {
      swap_endian_inplace<T>(ptr + i);
    }
  }

  return matrix;
}

/**
@brief write a span of data to a binary file.

@tparam T The type of the elements in the span. Must be trivially copyable.
@param filename The name of the file to write to.
@param data A span consisting of the data to be written. The span provides a
view into a sequence of objects of type `T`.
*/
template <typename T> void to_bin(fs::path filename, std::span<const T> data) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file\n";
    return;
  }
  const std::streamsize datasize_bytes = data.size() * sizeof(T);
  file.write(
      reinterpret_cast<const char *>(data.data()), // NOLINT(*-reinterpret-cast)
      datasize_bytes);
}

} // namespace uspam::io
