/*
This module implements a data acquisition interface
*/
#pragma once

#ifdef ARPAM_HAS_ALAZAR

#include "Common.hpp"
#include "RFBuffer.hpp"
#include <AlazarApi.h>
#include <QString>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace daq {

namespace fs = std::filesystem;

// Get DAQ board information
std::string getDAQInfo();

class DAQ {
public:
  explicit DAQ(std::shared_ptr<RFBuffer<ArpamFloat>> buffer)
      : m_buffer(std::move(buffer)) {}

  DAQ(const DAQ &) = delete;
  DAQ(DAQ &&) = delete;
  DAQ operator=(const DAQ &) = delete;
  DAQ operator=(DAQ &&) = delete;

  ~DAQ();

  // Initialize the DAQ board, including setting the clock, trigger, channel...
  [[nodiscard]] bool initHardware() noexcept;
  [[nodiscard]] bool isInitialized() const noexcept { return board != nullptr; }

  // Must be called before acquisition.
  // Allocates resources and primes the board
  [[nodiscard]] bool prepareAcquisition() noexcept;

  // Acquire `buffersToAcquire` buffers (BScans)
  [[nodiscard]] bool
  acquire(int buffersToAcquire, int indexOffset = 0,
          const std::function<void()> &startCallback = {}) noexcept;

  // Clean up resources allocated by "prepareAcquisition"
  void finishAcquisition() noexcept {
    if (m_fs.is_open()) {
      // Close file handle
      m_fs.close();
    }
  }

  // Set whether to save raw data or not.
  void setSaveData(bool save) noexcept { m_saveData = save; }

  void setSavedir(fs::path savedir) noexcept { m_savedir = std::move(savedir); }

  [[nodiscard]] auto &binpath() const noexcept { return m_lastBinfile; }

  [[nodiscard]] auto &errMsg() const noexcept { return m_errMsg; }

private:
  // RF Buffer
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  // Control states
  std::atomic<bool> shouldStopAcquiring{false};
  std::atomic<bool> acquiringData{false};

  // Alazar board handle
  void *board{};

  // Alazar Buffers
  std::array<std::span<uint16_t>, 20> buffers{};

  // No pre-trigger samples in NPT mode
  const U32 preTriggerSamples = 0;
  // Number of post trigger samples per record
  const U32 postTriggerSamples = 8192;

  // Number of samples total
  constexpr U32 samplesPerRecord() const noexcept {
    return preTriggerSamples + postTriggerSamples;
  }

  // Number of records per DMA buffer
  const U32 recordsPerBuffer = 1000;

  // Channels to capture
  const U32 channelMask = CHANNEL_A;
  const int channelCount = 1;

  // Sampling rate
  double samplesPerSec = 0.0;

  // File pointer
  bool m_saveData{true};
  std::fstream m_fs;
  fs::path m_savedir{"F:/ARPAM/"};
  fs::path m_lastBinfile;

  // Last error message
  QString m_errMsg;
};

} // namespace daq

#endif // ARPAM_HAS_ALAZAR
