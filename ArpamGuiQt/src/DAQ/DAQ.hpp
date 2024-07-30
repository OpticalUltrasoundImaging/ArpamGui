/*

This module implements a data acquisition interface

*/
#pragma once
#include <QObject>
#include <array>
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
  void cleanupHardware();

  bool startAcquisition();
  void stopAcquisition();

  void getBScan();

private:
  void allocateBuffers();
  void deallocateBuffers();

  // Alazar board handle
  void *board{};

  // Buffers
  std::array<uint16_t *, 4> buffers;

  double samplesPerSec = 0.0;

  // File pointer
  bool saveData{false};
  FILE *fpData{};

  // control conditions
  bool shouldStopAcquiring{false};
};

} // namespace daq

#endif
