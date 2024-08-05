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
#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace daq {
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

  bool initHardware();
  bool initialized() const { return board != nullptr; }
  bool isAcquiring() const { return acquiringData; };

  void getBScan();

signals:
  void messageBox(QString);

  void initHardwareSuccessful();

  void acquisitionStarted();
  void acquisitionStopped();

public slots:

  // 1. Allocates the alazar DMA buffers
  // 2. Opens file pointer to write bin data
  void prepareAcquisition();

  bool startAcquisition(int buffersToAcquire);
  void stopAcquisition();

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
  bool saveData{true};
  std::fstream m_fs;
};

} // namespace daq

#endif // ARPAM_HAS_ALAZAR
