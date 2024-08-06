#pragma once

#include "uspam/ioParams.hpp"
#include <algorithm>
#include <armadillo>
#include <bit>
#include <cassert>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace uspam::io {

// Swap the endianness of a value inplace
template <typename T> void swap_endian_inplace(T *val) {
  char *ptr = reinterpret_cast<char *>(val); // NOLINT
  std::reverse(ptr, ptr + sizeof(T));        // NOLINT
}

// Function to convert matrix using OpenCV's cv::parallel_for_
template <typename Tin, typename Tout>
void parallel_convert(const arma::Mat<Tin> &input, arma::Mat<Tout> &output) {
  output.set_size(input.n_rows, input.n_cols);
  cv::parallel_for_(cv::Range(0, input.n_cols), [&](const cv::Range &range) {
    for (int col = range.start; col < range.end; ++col) {
      const auto *inptr = input.colptr(col);
      auto *outptr = output.colptr(col);
      for (int i = 0; i < input.n_rows; ++i) {
        // NOLINTNEXTLINE(*-pointer-arithmetic)
        outptr[i] = static_cast<Tout>(inptr[i]);
      }
    }
  });
}

// Function to convert matrix using OpenCV's cv::parallel_for_
// With scaline by alpha and beta like cv::Mat::convertTo
template <typename Tin, typename Tout>
void parallel_convert_(const arma::Mat<Tin> &input, arma::Mat<Tout> &output,
                       const Tout alpha = 1, const Tout beta = 0) {
  const int rows = input.n_rows;
  const int cols = input.n_cols;
  output.set_size(input.n_rows, input.n_cols);
  cv::parallel_for_(cv::Range(0, cols), [&](const cv::Range &range) {
    for (int col = range.start; col < range.end; ++col) {
      const auto *inptr = input.colptr(col);
      auto *outptr = output.colptr(col);
      for (int i = 0; i < rows; ++i) {
        // NOLINTNEXTLINE(*-pointer-arithmetic)
        outptr[i] = cv::saturate_cast<Tout>(inptr[i]) * alpha + beta;
      }
    }
  });
}

template <typename Tin, typename Tout>
void copyRFWithScaling(Tin *bufIn, int sizeIn, arma::Mat<Tout> &rf,
                       int alinesPerBScan, int samplesPerAScan = RF_ALINE_SIZE)
  requires(std::is_floating_point_v<Tout>)
{
  assert(samplesPerAScan * alinesPerBScan * sizeof(Tin) == sizeIn);

  if (rf.n_rows != samplesPerAScan || rf.n_cols != alinesPerBScan) {
    rf.set_size(RF_ALINE_SIZE, alinesPerBScan);
  }

  if constexpr (std::is_same_v<Tin, Tout>) {
    // type stored in bin is the same type as the buffer give. Use directly
    // NOLINTNEXTLINE(*-reinterpret-cast)
    std::copy(bufIn, bufIn + sizeIn, rf.memptr());

  } else {
    // Type stored in bin different from the given buffer.
    // Construct temporary arma mat that shares memory with the input buffer
    arma::Mat<Tin> inMat(bufIn, samplesPerAScan, alinesPerBScan, false, true);

    // Convert from uint16_t to Float, also scale from uint16_t space to
    // voltage [-1, 1]
    constexpr Tout alpha =
        static_cast<Tout>(1) / static_cast<Tout>(1 << 15); // 1 / (2**15)
    constexpr Tout beta = -1;
    parallel_convert_<Tin, Tout>(inMat, rf, alpha, beta);
  }
}

template <typename Tin, typename Tout>
void copyRFWithScaling(const arma::Mat<Tin> &bufIn, arma::Mat<Tout> &rf)
  requires(std::is_floating_point_v<Tout>)
{
  if (rf.n_rows != bufIn.n_rows || rf.n_cols != bufIn.n_cols) {
    rf.set_size(bufIn.n_rows, bufIn.n_cols);
  }

  if constexpr (std::is_same_v<Tin, Tout>) {
    // type stored in bin is the same type as the buffer give
    // Trivial copy
    rf = bufIn;

  } else {
    // Type stored in bin different from the given buffer.

    // Convert from uint16_t to Float, also scale from uint16_t space to
    // voltage [-1, 1]
    constexpr Tout alpha =
        static_cast<Tout>(1) / static_cast<Tout>(1 << 15); // 1 / (2**15)
    constexpr Tout beta = -1;
    parallel_convert_<Tin, Tout>(bufIn, rf, alpha, beta);
  }
}

inline bool isPrefix(std::string_view prefix, const std::string_view str) {
  if (prefix.size() > str.size()) {
    return false;
  }
  return std::equal(prefix.begin(), prefix.end(), str.begin());
}

template <typename TypeInBin> class BinfileLoader {
public:
  BinfileLoader() = default;
  BinfileLoader(const IOParams &ioparams, const fs::path filename,
                int alinesPerBscan = NUM_ALINES_DETAULT) {
    open(filename);
    setParams(ioparams, alinesPerBscan);
  }

  void setParams(const IOParams &ioparams,
                 int alinesPerBscan = NUM_ALINES_DETAULT) {
    this->m_byteOffset = ioparams.byte_offset;
    this->m_alinesPerBscan = alinesPerBscan;
  }

  void open(const fs::path &filename) {
    // Open file and seek to end
    m_file = std::ifstream(filename, std::ios::binary | std::ios::ate);
    if (!m_file.is_open()) {
      throw std::runtime_error(
          std::string("[BinfileLoader] Failed to open file ") +
          filename.generic_string());
    }
    m_scanIdx = 0;
    const std::streamsize fsize = m_file.tellg();

    if (isPrefix("ARPAM", filename.stem().string())) {
      this->m_byteOffset = 0;
    }

    m_numScans = (fsize - this->m_byteOffset) / scanSizeBytes();
    m_file.seekg(this->m_byteOffset, std::ios::beg);
  }

  [[nodiscard]] bool isOpen() const { return m_file.is_open(); }
  void close() {
    m_file.close();
    m_numScans = 0;
    m_scanIdx = 0;
  }

  // (bytes) Raw RF size of one PAUS scan
  auto scanSizeBytes() const {
    return RF_ALINE_SIZE * m_alinesPerBscan * sizeof(TypeInBin);
  }

  auto size() const { return isOpen() ? m_numScans : 0; }

  auto alinesPerBscan() const { return m_alinesPerBscan; }

  auto idx() const { return m_scanIdx; }
  inline void setIdx(int idx) {
    assert(idx >= 0 && idx < m_numScans);
    m_scanIdx = idx;
  }

  bool hasMoreScans() const { return isOpen() && m_scanIdx < m_numScans; }

  /**
  Getters
  */

  template <typename T> bool get(arma::Mat<T> &rf) {
    if (!isOpen()) [[unlikely]] {
      return false;
    }

    assert(m_scanIdx < m_numScans);

    const auto sizeBytes = scanSizeBytes();
    const auto start_pos = this->m_byteOffset + sizeBytes * m_scanIdx;
    m_file.seekg(start_pos, std::ios::beg);

    if (rf.n_rows != RF_ALINE_SIZE || rf.n_cols != m_alinesPerBscan) {
      rf.set_size(RF_ALINE_SIZE, m_alinesPerBscan);
    }

    if constexpr (std::is_same_v<T, TypeInBin>) {
      // type stored in bin is the same type as the buffer give. Use directly
      // NOLINTNEXTLINE(*-reinterpret-cast)
      return !m_file.read(reinterpret_cast<char *>(rf.memptr()), sizeBytes);

    } else {
      // Type stored in bin different from the given buffer.
      // Read into readBuffer first then convert
      if (m_readBuffer.n_rows != RF_ALINE_SIZE ||
          m_readBuffer.n_cols != m_alinesPerBscan) {
        m_readBuffer.resize(RF_ALINE_SIZE, m_alinesPerBscan);
      }

      // Read file
      // NOLINTNEXTLINE(*-reinterpret-cast)
      if (m_file.read(reinterpret_cast<char *>(m_readBuffer.memptr()),
                      sizeBytes)) {

        // Convert from uint16_t to Float, also scale from uint16_t space to
        // voltage [-1, 1]
        constexpr T alpha =
            static_cast<T>(1) / static_cast<T>(1 << 15); // 1 / (2**15)
        constexpr T beta = -1;
        parallel_convert_<TypeInBin, T>(m_readBuffer, rf, alpha, beta);

        return true;
      }
      return false;
    }
  }

  template <typename T> inline bool get(arma::Mat<T> &rf, int idx) {
    setIdx(idx);
    return get(rf);
  }

  template <typename T> auto get() -> arma::Mat<T> {
    arma::Mat<T> out;
    get(out);
    return out;
  }

  template <typename T> auto get(int idx) -> arma::Mat<T> {
    setIdx(idx);
    return get<T>();
  }

  auto getNext(arma::Mat<TypeInBin> &rf) -> bool {
    if (!isOpen()) [[unlikely]] {
      return false;
    }

    const auto ret = get(rf);
    m_scanIdx++;
    return ret;
  }

private:
  std::ifstream m_file;
  int m_byteOffset = 0;
  int m_numScans = 0;
  int m_alinesPerBscan = 0;
  int m_scanIdx = 0;

  arma::Mat<TypeInBin> m_readBuffer;
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
template <typename T>
void to_bin(const fs::path &filename, std::span<const T> data) {
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
