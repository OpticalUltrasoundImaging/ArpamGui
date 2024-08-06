#include "Motor/NI.hpp"

#ifdef ARPAM_HAS_NI

#include "NIDAQmx.h"
#include <span>

// NOLINTBEGIN(*-do-while, *-pointer-decay)

#define DAQMX_CALL(fn)                                                         \
  do {                                                                         \
    ret = fn;                                                                  \
    if (DAQmxFailed(ret)) {                                                    \
      success = false;                                                         \
      DAQmxGetExtendedErrorInfo(errBuf, sizeof(errBuf));                       \
      qCritical("Error: %s failed -- Code %d, %s\n", #fn, ret, errBuf);        \
    }                                                                          \
  } while (0)

std::string motor::getNIDAQInfo() {
  uInt32 vMajor = 0;
  uInt32 vMinor = 0;
  uInt32 vPatch = 0;
  int32_t ret = 0;
  char errBuf[1024] = {'\0'};
  bool success = true;

  DAQMX_CALL(DAQmxGetSysNIDAQMajorVersion(&vMajor));
  if (!success) {
    return errBuf;
  }
  DAQMX_CALL(DAQmxGetSysNIDAQMinorVersion(&vMinor));
  if (!success) {
    return errBuf;
  }
  DAQMX_CALL(DAQmxGetSysNIDAQUpdateVersion(&vPatch));
  if (!success) {
    return errBuf;
  }

  auto s = fmt::format("NIDAQ version {}.{}.{}\n", vMajor, vMinor, vPatch);
  return s;
}

namespace {

void squareWave(std::span<double> data, int sampleRate, double amplitude,
                double offset, double dutyCycle = 0.5) {
  // Generate square wave
  const auto samples = data.size();
  const auto period = static_cast<int>(samples / sampleRate);
  const auto periodOn = static_cast<int>(0.5 * period);
  int i = 0;
  while (i < samples) {
    int currPeriod = 0;
    for (; i < samples && currPeriod < periodOn; ++i, ++currPeriod) {
      data[i] = offset + amplitude;
    }
    for (; i < samples && currPeriod < period; ++i, ++currPeriod) {
      data[i] = offset - amplitude;
    }
  }
}

} // namespace

bool motor::MotorNI::prepareMove() {
  errBuf[0] = '\0';
  bool success = true;

  // Create a Task and Virtual Channels
  DAQMX_CALL(DAQmxCreateTask("MoveMotor", &taskHandle));
  if (success) {

    // Create AO voltage channel
    const char *physicalChannel = "Dev1/ao0";
    constexpr float64 minVal = 0.0;
    constexpr float64 maxVal = 5.0;
    constexpr int32 units = DAQmx_Val_Volts;
    DAQMX_CALL(DAQmxCreateAOVoltageChan(taskHandle, physicalChannel, "", minVal,
                                        maxVal, units, nullptr));
  }

  // Sampling options
  constexpr float64 sampleRate =
      1e6; //  sampling rate in samples/sec per channel
  constexpr uInt64 sampsPerChan =
      1e6; // num samples to acquire/generate for each channel

  if (success) {
    constexpr auto activeEdge = DAQmx_Val_Rising;
    constexpr auto sampleMode = DAQmx_Val_FiniteSamps;
    DAQMX_CALL(DAQmxCfgSampClkTiming(taskHandle, "", sampleRate, activeEdge,
                                     sampleMode, sampsPerChan));
  }

  if (success) {
    const float32 timeout =
        10; // Time (s) to wait for the function to read/write all the samples
    const bool32 autoStart = 0; // Whether or not this function automatically
                                // starts the task if you do not start it
    const bool32 dataLayout =
        DAQmx_Val_GroupByChannel; // How data is arranged. Interleaved or
                                  // noninterleaved
    if (data.size() != sampsPerChan) {
      data.resize(sampsPerChan);
      squareWave(data, 1600, 2.5, 2.5, 0.5);
    }

    DAQMX_CALL(DAQmxWriteAnalogF64(taskHandle, sampsPerChan, autoStart, timeout,
                                   dataLayout, data.data(), nullptr, nullptr));
  }
  return success;
}

bool motor::MotorNI::startMoveAsync() {
  bool success = true;
  DAQMX_CALL(DAQmxStartTask(taskHandle));
  return success;
}

bool motor::MotorNI::waitUntilMoveEnds() {
  bool success = true;

  DAQMX_CALL(DAQmxWaitUntilTaskDone(taskHandle, 2));

  // Stop and Clear the Task.
  if (taskHandle != 0) {
    DAQmxStopTask(taskHandle);
    DAQmxClearTask(taskHandle);
  }
  return success;
}

void motor::MotorNI::abortMove() {
  if (taskHandle != 0) {
    DAQmxStopTask(taskHandle);
    DAQmxClearTask(taskHandle);
  }
}

bool motor::MotorNI::moveBlocking() {
  bool success = true;

  // Create a Task and Virtual Channels
  DAQMX_CALL(DAQmxCreateTask("MoveMotor", &taskHandle));

  if (success) {
    // Create AO voltage channel
    const char *physicalChannel = "Dev1/ao0";
    constexpr float64 minVal = 0.0;
    constexpr float64 maxVal = 5.0;
    constexpr int32 units = DAQmx_Val_Volts;
    DAQMX_CALL(DAQmxCreateAOVoltageChan(taskHandle, physicalChannel, "", minVal,
                                        maxVal, units, nullptr));
  }

  // Sampling options
  constexpr float64 sampleRate =
      1e6; //  sampling rate in samples/sec per channel
  constexpr uInt64 sampsPerChan =
      1e6; // num samples to acquire/generate for each channel
  if (success) {
    constexpr auto activeEdge = DAQmx_Val_Rising;
    constexpr auto sampleMode = DAQmx_Val_FiniteSamps;

    DAQMX_CALL(DAQmxCfgSampClkTiming(taskHandle, "", sampleRate, activeEdge,
                                     sampleMode, sampsPerChan));
  }

  /**
  Write
  */

  const float32 timeout =
      10; // Time (s) to wait for the function to read/write all the samples

  const bool32 autoStart = 0; // Whether or not this function automatically
                              // starts the task if you do not start it

  const bool32 dataLayout =
      DAQmx_Val_GroupByChannel; // How data is arranged. Interleaved or
                                // noninterleaved

  if (data.size() != sampsPerChan) {
    data.resize(sampsPerChan);
    squareWave(data, 1600, 2.5, 2.5, 0.5);
  }

  if (success) {
    DAQMX_CALL(DAQmxWriteAnalogF64(taskHandle, sampsPerChan, autoStart, timeout,
                                   dataLayout, data.data(), nullptr, nullptr));
  }

  /**
  Start
  */
  if (success) {
    DAQMX_CALL(DAQmxStartTask(taskHandle));
    DAQMX_CALL(DAQmxWaitUntilTaskDone(taskHandle, 10));
  }

  // Stop and Clear the Task.
  if (taskHandle != 0) {
    DAQmxStopTask(taskHandle);
    DAQmxClearTask(taskHandle);
  }
  return success;
}

bool motor::MotorNI::setDirection(Direction direction) {
  bool success{true};
  void *taskHandle{}; // Use our own taskHandle here to be safe from interfering
                      // with async move

  // Create a Task and Virtual Channels
  DAQMX_CALL(DAQmxCreateTask("SetMotorDirection", &taskHandle));

  // Create DIO channel
  if (success) {
    DAQMX_CALL(DAQmxCreateDOChan(taskHandle, "Dev1/port1", "",
                                 DAQmx_Val_ChanForAllLines));
  }

  if (success) {
    DAQMX_CALL(DAQmxStartTask(taskHandle));
  }

  const float32 timeout =
      10; // Time (s) to wait for the function to read/write all the samples
  const bool32 autoStart = 1;

  const bool32 dataLayout =
      DAQmx_Val_GroupByChannel; // How data is arranged. Interleaved or
  // noninterleaved
  const uInt32 data = (direction == Direction::CLOCKWISE) ? 0xffffffff : 0x0;

  if (success) {
    DAQMX_CALL(DAQmxWriteDigitalU32(taskHandle, 1, autoStart, timeout,
                                    dataLayout, &data, nullptr, nullptr));
  }

  if (taskHandle != 0) {
    DAQmxStopTask(taskHandle);
    DAQmxClearTask(taskHandle);
  }

  return success;
}

// NOLINTEND(*-do-while, *-pointer-decay)

#endif // ARPAM_HAS_NI