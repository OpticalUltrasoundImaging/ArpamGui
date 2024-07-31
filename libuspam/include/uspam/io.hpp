#pragma once

#include "uspam/ioParams.hpp"
#include <algorithm>
#include <armadillo>
#include <bit>
#include <cassert>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <span>
#include <string>
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
  output.set_size(input.n_rows, input.n_cols);
  cv::parallel_for_(cv::Range(0, input.n_cols), [&](const cv::Range &range) {
    for (int col = range.start; col < range.end; ++col) {
      const auto *inptr = input.colptr(col);
      auto *outptr = output.colptr(col);
      for (int i = 0; i < input.n_rows; ++i) {
        // NOLINTNEXTLINE(*-pointer-arithmetic)
        outptr[i] = cv::saturate_cast<Tout>(inptr[i]) * alpha + beta;
      }
    }
  });
}

template <typename TypeInBin> class BinfileLoader {
private:
  std::ifstream file;
  int byteOffset = 0;
  int numScans = 0;
  int alinesPerBscan = 0;
  int currScanIdx = 0;
  mutable std::mutex mtx;

  arma::Mat<TypeInBin> readBuffer;

public:
  BinfileLoader() = default;
  BinfileLoader(const IOParams &ioparams, const fs::path filename,
                int alinesPerBscan = NUM_ALINES_DETAULT) {
    open(filename);
    setParams(ioparams, alinesPerBscan);
  }

  void setParams(const IOParams &ioparams,
                 int alinesPerBscan = NUM_ALINES_DETAULT) {
    this->byteOffset = ioparams.byte_offset;
    this->alinesPerBscan = alinesPerBscan;
  }

  void open(const fs::path &filename) {
    // Open file and seek to end
    file = std::ifstream(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error(
          std::string("[BinfileLoader] Failed to open file ") +
          filename.generic_string());
    }
    const std::streamsize fsize = file.tellg();
    numScans = (fsize - this->byteOffset) / scanSizeBytes();
    file.seekg(this->byteOffset, std::ios::beg);
  }

  void close() { file.close(); }
  bool isOpen() const { return file.is_open(); }

  // (bytes) Raw RF size of one PAUS scan
  auto scanSizeBytes() const {
    return RF_ALINE_SIZE * alinesPerBscan * sizeof(TypeInBin);
  }

  auto size() const { return isOpen() ? numScans : 0; }

  void setCurrIdx(int idx) {
    if (!isOpen()) [[unlikely]] {
      return;
    }

    std::lock_guard lock(mtx);
    assert(idx >= 0 && idx < numScans);
    currScanIdx = idx;
  }

  bool hasMoreScans() const {
    if (!isOpen()) [[unlikely]] {
      return false;
    }

    std::lock_guard lock(mtx);
    return currScanIdx < numScans;
  }

  auto setCurrIndex(int idx) { currScanIdx = idx; }

  template <typename T> bool get(arma::Mat<T> &rf) {
    if (!isOpen()) [[unlikely]] {
      return false;
    }

    std::lock_guard lock(mtx);
    assert(currScanIdx < numScans);

    const auto sizeBytes = scanSizeBytes();
    const auto start_pos = this->byteOffset + sizeBytes * currScanIdx;
    file.seekg(start_pos, std::ios::beg);

    if (rf.n_rows != RF_ALINE_SIZE || rf.n_cols != alinesPerBscan) {
      rf.set_size(RF_ALINE_SIZE, alinesPerBscan);
    }

    if constexpr (std::is_same_v<T, TypeInBin>) {
      // type stored in bin is the same type as the buffer give. Use directly
      // NOLINTNEXTLINE(*-reinterpret-cast)
      return !file.read(reinterpret_cast<char *>(rf.memptr()), sizeBytes);

    } else {
      // Type stored in bin different from the given buffer.
      // Read into .readBuffer first then convert
      if (readBuffer.n_rows != RF_ALINE_SIZE ||
          readBuffer.n_cols != alinesPerBscan) {
        readBuffer.resize(RF_ALINE_SIZE, alinesPerBscan);
      }

      // Read file
      // NOLINTNEXTLINE(*-reinterpret-cast)
      if (file.read(reinterpret_cast<char *>(readBuffer.memptr()), sizeBytes)) {

        // Convert from uint16_t to Float, also scale from uint16_t space to
        // voltage [-1, 1]
        constexpr T alpha =
            static_cast<T>(1) / static_cast<T>(1 << 15); // 1 / (2**15)
        constexpr T beta = -1;
        parallel_convert_<TypeInBin, T>(readBuffer, rf, alpha, beta);

        return true;
      }
      return false;
    }
  }

  template <typename T> inline bool get(arma::Mat<T> &rf, int idx) {
    setCurrIdx(idx);
    return get<T>(rf);
  }

  template <typename T> auto get() -> arma::Mat<T> {
    arma::Mat<T> out;
    get(out);
    return out;
  }

  template <typename T> auto get(int idx) -> arma::Mat<T> {
    setCurrIdx(idx);
    return get<T>();
  }

  auto getNext(arma::Mat<TypeInBin> &rfStorage) -> bool {
    if (!isOpen()) [[unlikely]] {
      return false;
    }

    const auto ret = get(rfStorage);
    std::lock_guard lock(mtx);
    currScanIdx++;
    return ret;
  }

  auto getAlinesPerBscan() const { return alinesPerBscan; }
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
