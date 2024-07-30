/*

This module implements a data acquisition interface

*/
#pragma once
#include <QObject>
#include <QString>
#include <array>
#include <atomic>
#include <cstdio>
#include <string>

#ifdef ARPAM_HAS_ALAZARTECH

namespace daq {
// Get DAQ board information
std::string getDAQInfo();

class DAQ : public QObject {
  Q_OBJECT

public:
  DAQ();

  DAQ(const DAQ &) = delete;
  DAQ(DAQ &&) = delete;
  DAQ operator=(const DAQ &) = delete;
  DAQ operator=(DAQ &&) = delete;

  ~DAQ();

  bool initHardware();
  bool isInitialized();
  bool isAcquiring() { return acquiringData; };

  void getBScan();

signals:
  void messageBox(QString);
  void acquisitionStarted();
  void acquisitionStopped();

public slots:
  bool startAcquisition();
  void stopAcquisition();

private:
  // Control states
  std::atomic<bool> shouldStopAcquiring{false};
  std::atomic<bool> acquiringData{false};

  // Alazar board handle
  void *board{};

  // Buffers
  std::array<uint16_t *, 4> buffers;

  double samplesPerSec = 0.0;

  // File pointer
  bool saveData{true};
  FILE *fpData{};
};

} // namespace daq

#endif
