#include "DAQ/DAQ.hpp"

#include "AlazarApi.h"
#include "AlazarCmd.h"
#include "AlazarError.h"
#include <fmt/format.h>
#include <sstream>
#include <string>

namespace {

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

  RETURN_CODE retCode{};
  U32 samplesPerChannel{};
  BYTE bitsPerSample{};
  retCode = AlazarGetChannelInfo(handle, &samplesPerChannel, &bitsPerSample);
  if (retCode != ApiSuccess) {
    return fmt::format("Error: AlazarGetChannelInfo failed -- {}\n",
                       AlazarErrorToText(retCode));
  }

  U32 aspocType{};
  retCode = AlazarQueryCapability(handle, ASOPC_TYPE, 0, &aspocType);
  if (retCode != ApiSuccess) {
    return fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                       AlazarErrorToText(retCode));
  }

  BYTE fpgaMajor{};
  BYTE fpgaMinor{};
  retCode = AlazarGetFPGAVersion(handle, &fpgaMajor, &fpgaMinor);
  if (retCode != ApiSuccess) {
    return fmt::format("Error: AlazarGetFPGAVersion failed -- {}.\n",
                       AlazarErrorToText(retCode));
  }

  BYTE cpldMajor{};
  BYTE cpldMinor{};
  retCode = AlazarGetCPLDVersion(handle, &cpldMajor, &cpldMinor);
  if (retCode != ApiSuccess) {
    return fmt::format("Error: AlazarGetCPLDVersion failed -- {}.\n",
                       AlazarErrorToText(retCode));
  }

  U32 serialNumber{};
  retCode = AlazarQueryCapability(handle, GET_SERIAL_NUMBER, 0, &serialNumber);
  if (retCode != ApiSuccess) {
    return fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                       AlazarErrorToText(retCode));
  }

  U32 latestCalDate{};
  retCode =
      AlazarQueryCapability(handle, GET_LATEST_CAL_DATE, 0, &latestCalDate);
  if (retCode != ApiSuccess) {
    return fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                       AlazarErrorToText(retCode));
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
    retCode = AlazarQueryCapability(handle, GET_CPF_DEVICE, 0, &deviceType);
    if (retCode != ApiSuccess) {
      ss << fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                        AlazarErrorToText(retCode));
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
    retCode = AlazarQueryCapability(handle, GET_PCIE_LINK_SPEED, 0, &linkSpeed);
    if (retCode != ApiSuccess) {
      ss << fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                        AlazarErrorToText(retCode));
    }

    U32 linkWidth{};
    retCode = AlazarQueryCapability(handle, GET_PCIE_LINK_WIDTH, 0, &linkWidth);
    if (retCode != ApiSuccess) {
      ss << fmt::format("Error: AlazarQueryCapability failed -- {}.\n",
                        AlazarErrorToText(retCode));
    }

    ss << fmt::format("PCIe link speed = {} Gbps\n",
                      2.5 * linkSpeed); // NOLINT
    ss << fmt::format("PCIe link width = {} lanes\n", linkWidth);

    float fpgaTemperature_degreesC{};
    retCode = AlazarGetParameterUL(handle, CHANNEL_ALL, GET_FPGA_TEMPERATURE,
                                   (U32 *)&fpgaTemperature_degreesC);
    if (retCode != ApiSuccess) {
      ss << fmt::format("Error: AlazarGetParameterUL failed -- {}.\n",
                        AlazarErrorToText(retCode));
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
} // namespace daq