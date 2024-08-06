/*
This module implements a data acquisition interface
*/
#pragma once

#ifdef ARPAM_HAS_ALAZAR

#include "Common.hpp"
#include "RFBuffer.hpp"
#include <QObject>
#include <QString>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

namespace daq {

namespace fs = std::filesystem;

// Get DAQ board information
std::string getDAQInfo();

class DAQ : public QObject {
  Q_OBJECT

public:
  explicit DAQ(std::shared_ptr<RFBuffer<ArpamFloat>> buffer)
      : m_buffer(std::move(buffer)) {}

  DAQ(const DAQ &) = delete;
  DAQ(DAQ &&) = delete;
  DAQ operator=(const DAQ &) = delete;
  DAQ operator=(DAQ &&) = delete;

  ~DAQ();

  // Initialize the DAQ board, including setting the clock, trigger, channel...
  bool initHardware();

  bool initialized() const { return board != nullptr; }
  bool isAcquiring() const { return acquiringData; };

signals:
  void messageBox(QString);

  void initHardwareSuccessful();

  void acquisitionStarted();
  void acquisitionStopped();

  void finishedAcquiringBinfile(fs::path);

public slots:

  // Acquire `buffersToAcquire` buffers (BScans)
  bool startAcquisition(int buffersToAcquire, int indexOffset = 0);

  // Signal the acquisition thread to exit.
  void stopAcquisition() { shouldStopAcquiring = true; }

  void finishAcquisition() {
    if (m_fs.is_open()) {
      // Close file handle
      m_fs.close();

      emit finishedAcquiringBinfile(m_lastBinfile);
    }
  }

  // Set whether to save raw data or not.
  void setSaveData(bool save) { m_saveData = save; }

  void setSavedir(fs::path savedir) { m_savedir = std::move(savedir); }

private:
  // Buffer
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  // Control states
  std::atomic<bool> shouldStopAcquiring{false};
  std::atomic<bool> acquiringData{false};

  // Alazar board handle
  void *board{};

  // Buffers
  std::array<uint16_t *, 4> buffers{};

  double samplesPerSec = 0.0;

  // File pointer
  bool m_saveData{true};
  std::fstream m_fs;
  fs::path m_savedir;
  fs::path m_lastBinfile;
};

} // namespace daq

#endif // ARPAM_HAS_ALAZAR
