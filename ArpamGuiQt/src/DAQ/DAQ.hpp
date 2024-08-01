/*
This module implements a data acquisition interface
*/
#pragma once

#include "Common.hpp"
#include "RFBuffer.hpp"
#include <QObject>
#include <QString>
#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#ifdef ARPAM_HAS_ALAZARTECH

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
  void acquisitionStarted();
  void acquisitionStopped();

public slots:
  bool startAcquisition();
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
  FILE *fpData{};
};

} // namespace daq

#endif
