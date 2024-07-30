#include "DAQ/DAQ.hpp"

#include "AlazarApi.h"
#include "AlazarCmd.h"
#include "AlazarError.h"
#include <QtLogging>
#include <fmt/format.h>
#include <qlogging.h>
#include <sstream>
#include <string>

// NOLINTBEGIN(*-do-while)

namespace {

// Wrapper error handling for ATS-SDK call
// Need to define 1 variable before call:
// RETURN_CODE ret{ApiSuccess};
// bool success{true};
// NOLINTNEXTLINE(*-macro-usage)
#define ALAZAR_CALL(fn)                                                        \
  do {                                                                         \
    ret = fn;                                                                  \
    success = ret != ApiSuccess;                                               \
    if (success) {                                                             \
      qCritical("Error: %s failed -- %s\n", #fn, AlazarErrorToText(ret));      \
    }                                                                          \
  } while (0)

// NOLINTNEXTLINE(*-macro-usage)
#define RETURN_IF_FAIL()                                                       \
  if (ret != ApiSuccess) {                                                     \
    return false;                                                              \
  }

std::string GetSystemInfo(U32 systemId);
std::string GetBoardInfo(U32 systemId, U32 boardId);
bool IsPcieDevice(HANDLE handle);
const char *BoardTypeToText(int boardType);
bool HasCoprocessorFPGA(HANDLE handle);

std::string GetSystemInfo(U32 systemId) {
  std::stringstream ss;

  U32 boardCount = AlazarBoardsInSystemBySystemID(systemId);
  if (boardCount == 0) {
    ss << "Error: No boards found in system.\n";
    return ss.str();
  }

  HANDLE handle = AlazarGetSystemHandle(systemId);
  if (handle == nullptr) {
    ss << "Error: AlazarGetSystemHandle system failed.\n";
    return ss.str();
  }

  int boardType = AlazarGetBoardKind(handle);
  if (boardType == ATS_NONE || boardType >= ATS_LAST) {
    ss << fmt::format("Error: Unknown board type {}\n", boardType);
    return ss.str();
  }

  U8 driverMajor{};
  U8 driverMinor{};
  U8 driverRev{};
  auto retCode = AlazarGetDriverVersion(&driverMajor, &driverMinor, &driverRev);
  if (retCode != ApiSuccess) {
    ss << fmt::format("Error: AlazarGetDriverVersion failed -- {}\n",
                      AlazarErrorToText(retCode));
    return ss.str();
  }

  ss << fmt::format("System ID = {}\n", systemId);
  ss << fmt::format("Board type = {}\n", BoardTypeToText(boardType));
  ss << fmt::format("Board count = {}\n", boardCount);
  ss << fmt::format("Driver version = {}.{}.{}\n", driverMajor, driverMinor,
                    driverRev);

  // Display informataion about each board in this board system

  for (U32 boardId = 1; boardId <= boardCount; boardId++) {
    ss << "\n" << GetBoardInfo(systemId, boardId);
  }
  return ss.str();
}

std::string GetBoardInfo(U32 systemId, U32 boardId) {
  HANDLE handle = AlazarGetBoardBySystemID(systemId, boardId);
  if (handle == nullptr) {
    return fmt::format("Error: Open systemId {} boardId {} failed\n", systemId,
                       boardId);
  }

  RETURN_CODE ret{ApiSuccess};
  U32 samplesPerChannel{};
  BYTE bitsPerSample{};
  ret = AlazarGetChannelInfo(handle, &samplesPerChannel, &bitsPerSample);
  if (ret != ApiSuccess) {
    return fmt::format("Error: AlazarGetChannelInfo failed -- {}\n",
                       AlazarErrorToText(ret));
  }

  U32 aspocType{};
  ret = AlazarQueryCapability(handle, ASOPC_TYPE, 0, &aspocType);
  if (ret != ApiSuccess) {
    return fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                       AlazarErrorToText(ret));
  }

  BYTE fpgaMajor{};
  BYTE fpgaMinor{};
  ret = AlazarGetFPGAVersion(handle, &fpgaMajor, &fpgaMinor);
  if (ret != ApiSuccess) {
    return fmt::format("Error: AlazarGetFPGAVersion failed -- {}.\n",
                       AlazarErrorToText(ret));
  }

  BYTE cpldMajor{};
  BYTE cpldMinor{};
  ret = AlazarGetCPLDVersion(handle, &cpldMajor, &cpldMinor);
  if (ret != ApiSuccess) {
    return fmt::format("Error: AlazarGetCPLDVersion failed -- {}.\n",
                       AlazarErrorToText(ret));
  }

  U32 serialNumber{};
  ret = AlazarQueryCapability(handle, GET_SERIAL_NUMBER, 0, &serialNumber);
  if (ret != ApiSuccess) {
    return fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                       AlazarErrorToText(ret));
  }

  U32 latestCalDate{};
  ret = AlazarQueryCapability(handle, GET_LATEST_CAL_DATE, 0, &latestCalDate);
  if (ret != ApiSuccess) {
    return fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                       AlazarErrorToText(ret));
  }

  std::stringstream ss;
  ss << fmt::format("System ID = {}\n", systemId);
  ss << fmt::format("Board ID = {}\n", boardId);
  ss << fmt::format("Serial number = {}\n", serialNumber);
  ss << fmt::format("Bits per sample = {}\n", bitsPerSample);
  ss << fmt::format("Max samples per channel = {}\n", samplesPerChannel);
  ss << fmt::format("FPGA version = {}.{}\n", fpgaMajor, fpgaMinor);
  ss << fmt::format("CPLD version = {}.{}\n", cpldMajor, cpldMinor);
  ss << fmt::format("ASoPC signature = {:x}\n", aspocType);
  ss << fmt::format("Latest calibration date = {}\n", latestCalDate);

  if (HasCoprocessorFPGA(handle)) {
    // Display co-processor FPGA device type

    U32 deviceType{};
    ret = AlazarQueryCapability(handle, GET_CPF_DEVICE, 0, &deviceType);
    if (ret != ApiSuccess) {
      ss << fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                        AlazarErrorToText(ret));
      return ss.str();
    }

    const char *deviceName{};
    switch (deviceType) {
    case CPF_DEVICE_EP3SL50:
      deviceName = "EP3SL50";
      break;
    case CPF_DEVICE_EP3SE260:
      deviceName = "EP3SL260";
      break;
    default:
      deviceName = "Unknown";
      break;
    }
    ss << fmt::format("CPF Device = {}\n", deviceName);
  }

  if (IsPcieDevice(handle)) {
    // Display PCI Express link information

    U32 linkSpeed{};
    ret = AlazarQueryCapability(handle, GET_PCIE_LINK_SPEED, 0, &linkSpeed);
    if (ret != ApiSuccess) {
      ss << fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                        AlazarErrorToText(ret));
    }

    U32 linkWidth{};
    ret = AlazarQueryCapability(handle, GET_PCIE_LINK_WIDTH, 0, &linkWidth);
    if (ret != ApiSuccess) {
      ss << fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                        AlazarErrorToText(ret));
    }

    ss << fmt::format("PCIe link speed = {} Gbps\n",
                      2.5 * linkSpeed); // NOLINT
    ss << fmt::format("PCIe link width = {} lanes\n", linkWidth);

    float fpgaTemperature_degreesC{};
    ret = AlazarGetParameterUL(handle, CHANNEL_ALL, GET_FPGA_TEMPERATURE,
                               (U32 *)&fpgaTemperature_degreesC);
    if (ret != ApiSuccess) {
      ss << fmt::format("Error: AlazarGetParameterUL failed -- {}.\n",
                        AlazarErrorToText(ret));
      return ss.str();
    }

    ss << fmt::format("FPGA temperature = {} C\n", fpgaTemperature_degreesC);
  }

  return ss.str();
}

// Return true if board has PCIe host bus interface
bool IsPcieDevice(HANDLE handle) {
  U32 boardType = AlazarGetBoardKind(handle);
  return boardType >= ATS9462;
}

// Convert board type Id to text
const char *BoardTypeToText(int boardType) {
  // NOLINTNEXTLINE(*-default-case)
  switch (boardType) {
  case ATS850:
    return "ATS850";
  case ATS310:
    return "ATS310";
  case ATS330:
    return "ATS330";
  case ATS855:
    return "ATS855";
  case ATS315:
    return "ATS315";
  case ATS335:
    return "ATS335";
  case ATS460:
    return "ATS460";
  case ATS860:
    return "ATS860";
  case ATS660:
    return "ATS660";
  case ATS665:
    return "ATS665";
  case ATS9462:
    return "ATS9462";
  case ATS9870:
    return "ATS9870";
  case ATS9350:
    return "ATS9350";
  case ATS9325:
    return "ATS9325";
  case ATS9440:
    return "ATS9440";
  case ATS9351:
    return "ATS9351";
  case ATS9850:
    return "ATS9850";
  case ATS9625:
    return "ATS9625";
  case ATS9626:
    return "ATS9626";
  case ATS9360:
    return "ATS9360";
  case AXI9870:
    return "AXI9870";
  case ATS9370:
    return "ATS9370";
  case ATS9373:
    return "ATS9373";
  case ATS9416:
    return "ATS9416";
  case ATS9637:
    return "ATS9637";
  case ATS9120:
    return "ATS9120";
  case ATS9371:
    return "ATS9371";
  case ATS9130:
    return "ATS9130";
  case ATS9352:
    return "ATS9352";
  case ATS9353:
    return "ATS9353";
  case ATS9453:
    return "ATS9453";
  case ATS9146:
    return "ATS9146";
  case ATS9437:
    return "ATS9437";
  case ATS9618:
    return "ATS9618";
  case ATS9358:
    return "ATS9358";
  case ATS9872:
    return "ATS9872";
  case ATS9628:
    return "ATS9628";
  case ATS9364:
    return "ATS9364";
  }
  return "?";
}

// Return true if board has coprocessor FPGA
bool HasCoprocessorFPGA(HANDLE handle) {
  U32 boardType = AlazarGetBoardKind(handle);
  return (boardType == ATS9625 || boardType == ATS9626);
}

} // namespace

namespace daq {

std::string getDAQInfo() {
  std::stringstream ss;

  U8 sdkMajor{};
  U8 sdkMinor{};
  U8 sdkRevision{};
  auto retCode = AlazarGetSDKVersion(&sdkMajor, &sdkMinor, &sdkRevision);
  if (retCode != ApiSuccess) {
    return "";
  }

  auto systemCount = AlazarNumOfSystems();

  ss << fmt::format("Alazar SDK version = {}.{}.{}\n", sdkMajor, sdkMinor,
                    sdkRevision);
  ss << fmt::format("Alazar system count = {}\n", systemCount);

  if (systemCount < 1) {
    ss << fmt::format("No Alazar system found.\n");
  }

  else {
    for (U32 systemId = 1; systemId <= systemCount; ++systemId) {
      ss << GetSystemInfo(systemId);
    }
  }

  return ss.str();
}

DAQ::DAQ() { initHardware(); }

bool DAQ::initHardware() {
  /*
   * Initialize and configure board
   */

  RETURN_CODE ret{ApiSuccess};
  bool success{true};

  board = AlazarGetBoardBySystemID(1, 1);
  if (board == nullptr) {
    throw std::exception("Failed to initialize board");
  }

  // Specify the sample rate (see sample rate id below)
  samplesPerSec = 180e6;

  // Select clock parameters as required to generate this sample rate.
  //
  // For example: if samplesPerSec is 100.e6 (100 MS/s), then:
  // - select clock source INTERNAL_CLOCK and sample rate SAMPLE_RATE_100MSPS
  // - select clock source FAST_EXTERNAL_CLOCK, sample rate
  // SAMPLE_RATE_USER_DEF, and connect a
  //   100 MHz signal to the EXT CLK BNC connector.
  ALAZAR_CALL(AlazarSetCaptureClock(board, INTERNAL_CLOCK, SAMPLE_RATE_180MSPS,
                                    CLOCK_EDGE_RISING, 0));
  RETURN_IF_FAIL();

  // Select channel A input parameters as required
  ALAZAR_CALL(AlazarInputControlEx(board, CHANNEL_A, DC_COUPLING,
                                   INPUT_RANGE_PM_400_MV, IMPEDANCE_50_OHM));
  RETURN_IF_FAIL();

  // Select channel A bandwidth limit as required
  ALAZAR_CALL(AlazarSetBWLimit(board, CHANNEL_A, 0));
  RETURN_IF_FAIL();

  // Select channel B input parameters as required
  ALAZAR_CALL(AlazarInputControlEx(board, CHANNEL_B, DC_COUPLING,
                                   INPUT_RANGE_PM_400_MV, IMPEDANCE_50_OHM));
  RETURN_IF_FAIL();

  // : Select channel B bandwidth limit as required
  ALAZAR_CALL(AlazarSetBWLimit(board, CHANNEL_B, 0));
  RETURN_IF_FAIL();

  // Select trigger inputs and levels as required
  ALAZAR_CALL(AlazarSetTriggerOperation(board, TRIG_ENGINE_OP_J, TRIG_ENGINE_J,
                                        TRIG_CHAN_A, TRIGGER_SLOPE_POSITIVE,
                                        150, TRIG_ENGINE_K, TRIG_DISABLE,
                                        TRIGGER_SLOPE_POSITIVE, 128));
  RETURN_IF_FAIL();

  // Select external trigger parameters as required
  ALAZAR_CALL(AlazarSetExternalTrigger(board, DC_COUPLING, ETR_5V));
  RETURN_IF_FAIL();

  // Set trigger delay as required.
  const double triggerDelay_sec = 0;
  const U32 triggerDelay_samples =
      (U32)(triggerDelay_sec * samplesPerSec + 0.5);
  ALAZAR_CALL(AlazarSetTriggerDelay(board, triggerDelay_samples));
  RETURN_IF_FAIL();

  // Set trigger timeout as required.
  // NOTE:
  //
  // The board will wait for a for this amount of time for a trigger event. If
  // a trigger event does not arrive, then the board will automatically
  // trigger. Set the trigger timeout value to 0 to force the board to wait
  // forever for a trigger event.
  //
  // IMPORTANT:
  //
  // The trigger timeout value should be set to zero after appropriate trigger
  // parameters have been determined, otherwise the board may trigger if the
  // timeout interval expires before a hardware trigger event arrives.
  ALAZAR_CALL(AlazarSetTriggerTimeOut(board, 0));
  RETURN_IF_FAIL();

  // Configure AUX I/O connector as required
  ALAZAR_CALL(AlazarConfigureAuxIO(board, AUX_OUT_TRIGGER, 0));
  RETURN_IF_FAIL();

  return true;
}

bool DAQ::startAcquisition() {
  RETURN_CODE ret{ApiSuccess};
  bool success{true};

  // No pre-trigger samples in NPT mode
  U32 preTriggerSamples = 0;
  // Number of post trigger samples per record
  U32 postTriggerSamples = 8192;
  // Number of records per DMA buffer
  U32 recordsPerBuffer = 1000;

  // Total number of buffers to capture
  U32 buffersPerAcquisition = 100;

  // Channels to capture
  const U32 channelMask = CHANNEL_A;
  const int channelCount = 1;

  U8 bitsPerSample{};
  U32 maxSamplesPerChannel{};

  ALAZAR_CALL(
      AlazarGetChannelInfo(board, &maxSamplesPerChannel, &bitsPerSample));
  RETURN_IF_FAIL();

  auto bytesPerSample = (float)((bitsPerSample + 7) / 8);
  U32 samplesPerRecord = preTriggerSamples + postTriggerSamples;
  U32 bytesPerRecord =
      (U32)(bytesPerSample * samplesPerRecord +
            0.5); // 0.5 compensates for double to integer conversion
  U32 bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount;

  if (saveData) {
    fpData = fopen("data.bin", "wb");
    if (fpData == nullptr) {
      qCritical("Error: Unable to create data file.");
      return false;
    }
  }

  // Allocate memory for DMA buffers
  U32 bufferIdx = 0;
  for (; (bufferIdx < buffers.size()) && success; ++bufferIdx) {
    // Allocate page aligned memory
    buffers[bufferIdx] = AlazarAllocBufferU16(board, bytesPerBuffer);
    if (buffers[bufferIdx] == nullptr) {
      auto msg = fmt::format("Error: Alloc {} bytes failed\n", bytesPerBuffer);
      qCritical(msg.c_str());
      success = false;
    }
  }

  // Configure the record size
  ALAZAR_CALL(
      AlazarSetRecordSize(board, preTriggerSamples, postTriggerSamples));

  // Configure the board to make an NPT AutoDMA acquisition
  if (success) {
    U32 recordsPerAcquisition = recordsPerBuffer * buffersPerAcquisition;
    U32 admaFlags = ADMA_EXTERNAL_STARTCAPTURE | ADMA_NPT;
    ALAZAR_CALL(AlazarBeforeAsyncRead(
        board, channelMask, -(long)preTriggerSamples, samplesPerRecord,
        recordsPerBuffer, recordsPerAcquisition, admaFlags));
  }

  // Add the buffers to a list of buffers available to be filled by the board
  for (bufferIdx = 0; (bufferIdx < buffers.size()) && success; ++bufferIdx) {
    U16 *pBuffer = buffers[bufferIdx];
    ALAZAR_CALL(AlazarPostAsyncBuffer(board, pBuffer, bytesPerBuffer));
  }

  // Arm the board system to wait for a trigger event to begin acquisition
  ALAZAR_CALL(AlazarStartCapture(board));

  // Wait for each buffer to be filled, process the buffer, and re-post it to
  // the board.
  if (success) {
    qInfo("Capturing %d buffers ... press any key to abort\n",
          buffersPerAcquisition);

    U32 startTickCount = GetTickCount();
    U32 buffersCompleted = 0;
    INT64 bytesTransferred = 0;
    while (buffersCompleted < buffersPerAcquisition) {
      // TODO: Set a buffer timeout that is longer than the time
      //       required to capture all the records in one buffer.
      U32 timeout_ms = 5000;

      // Wait for the buffer at the head of the list of available buffers
      // to be filled by the board.
      bufferIdx = buffersCompleted % buffers.size();
      U16 *pBuffer = buffers[bufferIdx];

      ALAZAR_CALL(AlazarWaitAsyncBufferComplete(board, pBuffer, timeout_ms));

      if (success) {
        // The buffer is full and has been removed from the list
        // of buffers available for the board

        buffersCompleted++;
        bytesTransferred += bytesPerBuffer;

        // TODO: Process sample data in this buffer.

        // NOTE:
        //
        // While you are processing this buffer, the board is already filling
        // the next available buffer(s).
        //
        // You MUST finish processing this buffer and post it back to the
        // board before the board fills all of its available DMA buffers and
        // on-board memory.
        //
        // Records are arranged in the buffer as follows: R0A, R1A, R2A ...
        // RnA, R0B, R1B, R2B ... with RXY the record number X of channel Y
        //
        // Sample codes are unsigned by default. As a result:
        // - a sample code of 0x0000 represents a negative full scale input
        // signal.
        // - a sample code of 0x8000 represents a ~0V signal.
        // - a sample code of 0xFFFF represents a positive full scale input
        // signal.
        if (saveData) {
          // Write record to file
          size_t bytesWritten =
              fwrite(pBuffer, sizeof(BYTE), bytesPerBuffer, fpData);
          if (bytesWritten != bytesPerBuffer) {
            qCritical("Error: Write buffer %u failed -- %u\n", buffersCompleted,
                      GetLastError());
            success = false;
          }
        }
      }

      // Add the buffer to the end of the list of available buffers.
      ALAZAR_CALL(AlazarPostAsyncBuffer(board, pBuffer, bytesPerBuffer));

      // If the acquisition failed, exit the acquisition loop
      if (!success)
        break;

      // Check for condition to stop acquiring
      if (shouldStopAcquiring) {
        qInfo("Stopping acquisition\n");
        break;
      }

      // Display progress
      qInfo("Completed %u buffers\r", buffersCompleted);
    }

    // Display results
    double transferTime_sec = (GetTickCount() - startTickCount) / 1000.;
    qInfo("Capture completed in %.2lf sec\n", transferTime_sec);

    double buffersPerSec{};
    double bytesPerSec{};
    double recordsPerSec{};
    U32 recordsTransferred = recordsPerBuffer * buffersCompleted;
    if (transferTime_sec > 0.) {
      buffersPerSec = buffersCompleted / transferTime_sec;
      bytesPerSec = bytesTransferred / transferTime_sec;
      recordsPerSec = recordsTransferred / transferTime_sec;
    }

    qInfo("Captured %u buffers (%.4g buffers per sec)\n", buffersCompleted,
          buffersPerSec);
    qInfo("Captured %u records (%.4g records per sec)\n", recordsTransferred,
          recordsPerSec);
    qInfo("Transferred %I64d bytes (%.4g bytes per sec)\n", bytesTransferred,
          bytesPerSec);
  }

  // Abort the acquisition
  ALAZAR_CALL(AlazarAbortAsyncRead(board));

  // Free all memory allocated
  for (auto &pBuffer : buffers) {
    if (pBuffer != nullptr) {
      AlazarFreeBufferU16(board, pBuffer);
    }
    pBuffer = nullptr;
  }

  return true;
}

DAQ::~DAQ() {
  // Note: Alazar board handles don't need to be explicitly closed.

  // Free all memory allocated
  for (auto &pBuffer : buffers) {
    if (pBuffer != nullptr) {
      AlazarFreeBufferU16(board, pBuffer);
    }
    pBuffer = nullptr;
  }
}
} // namespace daq

// NOLINTEND(*-do-while)