#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

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

  static auto zeros_like(const PAUSpair &other) {
    return PAUSpair{arma::Mat<T>(other.PA.n_rows, other.PA.n_cols),
                    arma::Mat<T>(other.US.n_rows, other.US.n_cols)};
  }

  static auto empty_like(const PAUSpair &other) {
    return PAUSpair{
        arma::Mat<T>(other.PA.n_rows, other.PA.n_cols, arma::fill::none),
        arma::Mat<T>(other.US.n_rows, other.US.n_cols, arma::fill::none)};
  }
};

struct IOParams {
  int rf_size_PA;
  int rf_size_spacer;
  // No. of samples in US RF. Must be twice rf_size_PA
  int rf_size_US;

  int offset_US;
  // Number of points
  int offset_PA;

  // Byte offset at beginning of file.
  int byte_offset = 0;

  // System parameters from early 2024
  static inline IOParams system2024v1() {
    return IOParams{2650, 87, 5300, 350, 215, 1};
  }

  template <typename T> PAUSpair<T> allocate_split_pair(int alines_per_bscan=1000) const {
    arma::Mat<T> rfPA(this->rf_size_PA, alines_per_bscan, arma::fill::none);
    arma::Mat<T> rfUS(this->rf_size_US, alines_per_bscan, arma::fill::none);
    return {rfPA, rfUS};
  }

  template <typename T>
  auto split_rf_PAUS(const arma::Mat<T> &rf, PAUSpair<T> &split) const {
    const auto USstart = this->rf_size_PA + this->rf_size_spacer;
    const auto USend = USstart + this->rf_size_US;
    const auto offset_PA = this->offset_US / 2 + this->offset_PA;

    split.PA = rf.rows(0, this->rf_size_PA - 1);
    split.US = rf.rows(USstart, USend - 1);

    for (int i = 0; i < rf.n_cols; ++i) {
      {
        auto ptr = split.PA.colptr(i);
        std::rotate(ptr, ptr + this->offset_PA, ptr + split.PA.n_rows);
        // rfPA.rows(0, this->offset_PA - 1).zeros();
      }
      {
        auto ptr = split.US.colptr(i);
        std::rotate(ptr, ptr + this->offset_US, ptr + split.US.n_rows);
        // rfUS.rows(0, this->offset_US - 1).zeros();
      }
    }
  }

  template <typename T> auto split_rf_PAUS(const arma::Mat<T> &rf) const {
    const auto USstart = this->rf_size_PA + this->rf_size_spacer;
    const auto USend = USstart + this->rf_size_US;
    const auto offset_PA = this->offset_US / 2 + this->offset_PA;

    arma::Mat<T> rfPA = rf.rows(0, this->rf_size_PA - 1);
    arma::Mat<T> rfUS = rf.rows(USstart, USend - 1);

    for (int i = 0; i < rf.n_cols; ++i) {
      {
        auto ptr = rfPA.colptr(i);
        std::rotate(ptr, ptr + this->offset_PA, ptr + rfPA.n_rows);
        // rfPA.rows(0, this->offset_PA - 1).zeros();
      }
      {
        auto ptr = rfUS.colptr(i);
        std::rotate(ptr, ptr + this->offset_US, ptr + rfUS.n_rows);
        // rfUS.rows(0, this->offset_US - 1).zeros();
      }
    }

    return PAUSpair<T>{rfPA, rfUS};
  }

  template <typename T>
  auto load_rf(const fs::path &filename, int i, int nscans = 1,
               int alines_per_bscan = 1000) const -> arma::Mat<T> {
    // Only support nscans=1 for now
    assert(nscans >= 1);

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "Failed to open file\n";
      return {};
    }

    const auto scan_size =
        RF_ALINE_SIZE * alines_per_bscan * nscans * sizeof(T);
    const auto start_pos = this->byte_offset + scan_size * i;

    file.seekg(start_pos, std::ios::beg);

    // Read file
    arma::Mat<T> rf_storage(RF_ALINE_SIZE, alines_per_bscan, arma::fill::none);
    const auto matrix_size = RF_ALINE_SIZE * alines_per_bscan * sizeof(T);
    if (!file.read(reinterpret_cast<char *>(rf_storage.memptr()),
                   matrix_size)) {
      std::cerr << "Failed to read file\n";
      return {};
    }

    return rf_storage;
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
    int num_scans = (fsize - this->byte_offset) /
                    (RF_ALINE_SIZE * alines_per_bscan * sizeof(T));
    return num_scans;
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
