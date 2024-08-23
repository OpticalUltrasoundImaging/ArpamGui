#pragma once

#include "uspam/io.hpp"
#include "uspam/ioOldBin.hpp"
#include <array>
#include <filesystem>
#include <fmt/core.h>
#include <stdexcept>

namespace uspam::io {
namespace details {
const static inline std::array<const char *, 6> oldUSFilenames = {
    "NormalUS1.bin", "NormalUS2.bin", "NormalUS3.bin",
    "NormalUS4.bin", "NormalUS5.bin", "NormalUS6.bin"};

const static inline std::array<const char *, 6> oldPAFilenames = {
    "NormalPA1.bin", "NormalPA2.bin", "NormalPA3.bin",
    "NormalPA4.bin", "NormalPA5.bin", "NormalPA6.bin"};

} // namespace details

class OldBinLoader {
public:
  OldBinLoader() = default;

  void setSequencePath(const fs::path &sequence) {
    m_sequence.clear();

    if (!fs::exists(sequence)) {
      const auto msg =
          fmt::format("Sequence path does not exist: {}", sequence.string());
      throw std::runtime_error(msg);
    }

    /*
    The sequence folder should container 6 pairs of bin files.
    Check
    */
    for (const auto *const name : details::oldUSFilenames) {
      const auto path = sequence / name;
      if (!fs::exists(path)) {
        const auto msg =
            fmt::format("Seq directory incomplete: missing {}", name);
        throw std::runtime_error(msg);
      }
    }

    m_sequence = sequence;
  }

  [[nodiscard]] auto size() { return m_sequence.empty() ? 0 : 6; }

  template <typename T> bool get(arma::Mat<T> &rfUS, arma::Mat<T> &rfPA) {
    assert(m_scanIdx < size());
    const auto USpath = m_sequence / details::oldUSFilenames[m_scanIdx];
    const auto PApath = m_sequence / details::oldPAFilenames[m_scanIdx];

    rfUS = load_bin<uint16_t>(USpath, std::endian::big);
    rfPA = load_bin<uint16_t>(PApath, std::endian::big);
  }

private:
  fs::path m_sequence;
  int m_scanIdx = 0;
};
} // namespace uspam::io