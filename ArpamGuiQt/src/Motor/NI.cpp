#include "Motor/NI.hpp"

#ifdef ARPAM_HAS_NI

#include "NIDAQmx.h"

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

namespace motor {

std::string getNIDAQInfo() {
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

  return fmt::format("NIDAQ version {}.{}.{}\n", vMajor, vMinor, vPatch);
}

bool MotorNI::prepareMove(double speed, double rotations) {
  errBuf[0] = '\0';
  bool success = true;

  // Update driver square wave
  m_driver.setSpeed(speed);
  m_driver.setRotations(rotations);

  if (m_driver.needsUpdate() || taskHandle == nullptr) {
    // Only recreate task and recompute squareWave if needed
    const auto &squareWave = m_driver.squareWave();

    clearTask();

    // Create a Task and Virtual Channels
    DAQMX_CALL(DAQmxCreateTask("MoveMotor", &taskHandle));

    if (success) {
      // Create AO voltage channel
      const char *physicalChannel = "Dev1/ao0";
      constexpr float64 minVal = 0.0;
      constexpr float64 maxVal = 5.0;
      constexpr int32 units = DAQmx_Val_Volts;
      DAQMX_CALL(DAQmxCreateAOVoltageChan(taskHandle, physicalChannel, "",
                                          minVal, maxVal, units, nullptr));
    }

    // Sampling options for analogue out
    // sampling rate in samples/sec per channel
    // num samples to acquire/generate for each channel

    const uInt64 sampsPerChan = squareWave.size();

    if (success) {
      constexpr auto activeEdge = DAQmx_Val_Rising;
      constexpr auto sampleMode = DAQmx_Val_FiniteSamps;

      DAQMX_CALL(DAQmxCfgSampClkTiming(taskHandle, "", SampleRate, activeEdge,
                                       sampleMode, sampsPerChan));
    }

    /**
    Write
    */
    // Time (s) to wait for the function to read/write all the samples
    const float32 timeout = 10;
    // Whether this function automatically starts the task if you do not start
    // it
    const bool32 autoStart = 0;

    // How data is arranged. Interleaved or noninterleaved
    const bool32 dataLayout = DAQmx_Val_GroupByChannel;

    if (success) {
      DAQMX_CALL(DAQmxWriteAnalogF64(taskHandle, sampsPerChan, autoStart,
                                     timeout, dataLayout, squareWave.data(),
                                     nullptr, nullptr));
    }
  }

  return success;
}

bool MotorNI::startMoveAsync() {
  bool success = true;
  DAQMX_CALL(DAQmxStartTask(taskHandle));
  return success;
}

bool MotorNI::waitUntilMoveEnds() {
  bool success = true;

  DAQMX_CALL(
      DAQmxWaitUntilTaskDone(taskHandle, m_driver.approxMoveTimeSec() + 1));

  // Stop the Task.
  if (taskHandle != nullptr) {
    DAQmxStopTask(taskHandle);
  }

  return success;
}

void MotorNI::abortMove() {
  if (taskHandle != nullptr) {
    DAQmxStopTask(taskHandle);
    DAQmxClearTask(taskHandle);
  }
}

bool MotorNI::moveBlocking(double speed, double rotations) {
  bool success = prepareMove(speed, rotations);
  if (success) {
    success = startMoveAsync();
  }

  if (success) {
    success = waitUntilMoveEnds();
  }

  return success;
}

bool MotorNI::setDirection(Direction direction) {
  bool success{true};

  if (directionTaskHandle == nullptr) {
    // Create a Task and Virtual Channels
    DAQMX_CALL(DAQmxCreateTask("SetMotorDirection", &directionTaskHandle));

    // Create DIO channel
    if (success) {
      DAQMX_CALL(DAQmxCreateDOChan(directionTaskHandle, "Dev1/port1", "",
                                   DAQmx_Val_ChanForAllLines));
    }
  }

  if (success) {
    DAQMX_CALL(DAQmxStartTask(directionTaskHandle));
  }

  const float32 timeout =
      10; // Time (s) to wait for the function to read/write all the samples
  const bool32 autoStart = 1;

  const bool32 dataLayout =
      DAQmx_Val_GroupByChannel; // How data is arranged. Interleaved or
  // noninterleaved
  const uInt32 data = (direction == Direction::CLOCKWISE) ? 0xffffffff : 0x0;

  if (success) {
    DAQMX_CALL(DAQmxWriteDigitalU32(directionTaskHandle, 1, autoStart, timeout,
                                    dataLayout, &data, nullptr, nullptr));
  }

  DAQmxStopTask(directionTaskHandle);

  return success;
}

// NOLINTEND(*-do-while, *-pointer-decay)

} // namespace motor

#endif // ARPAM_HAS_NI
