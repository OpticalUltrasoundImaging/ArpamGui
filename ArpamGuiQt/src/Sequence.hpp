#pragma once

#include <AlazarCmd.h>
#include <cstdint>
#include <fmt/core.h>

namespace uspam {

// NOLINTBEGIN(*-magic-numbers)

struct RfMeta {
  // No. samples recorded (n samples to read from start)
  uint32_t size;
  // delay of samples (n zeros should be appended to the beginning of the read
  // RF during processing)
  uint32_t truncated;

  // Where in the saved Ascan the signal starts. Should filled in by a factory
  // function of the Sequence class below
  uint32_t offset_{};

  // No. samples that should be acquired
  [[nodiscard]] constexpr uint32_t acquired() const { return size + truncated; }
};

struct DepthInfo {
  float PAdepth;     // [mm] total depth
  float USdepth;     // [mm] total depth
  float PAtruncated; // [mm] truncated at the beginning
  float UStruncated; // [mm] truncated at the beginning

  [[nodiscard]] auto toString() const {
    return fmt::format("PA depth: {:.2f} mm\nUS depth: {:.2f} mm\n"
                       "PA truncated: {:.2f}\nUS truncated: {:.2f}\n",
                       PAdepth, USdepth, PAtruncated, UStruncated);
  }
};

struct DAQSequence {
  // Sampling freq
  // Currently support 180e6
  uint32_t SamplesPerSec;

  // Input range
  // valid entries currently are 800 (mV) and 1000 (mV)
  uint32_t InputRangePM_mV;

  uint32_t AscansPerBscan;
  uint32_t SamplesPerAscanAcquire;
  uint32_t SamplesPerAscanSave;

  RfMeta PA;
  RfMeta US;

  static constexpr DAQSequence make(const uint32_t SamplesPerSec,
                                    const uint32_t InputRangePM_mv,
                                    const uint32_t AscansPerBscan, RfMeta PA,
                                    RfMeta US) {

    PA.offset_ = 0;
    US.offset_ = PA.size;

    const uint32_t totalAcquire = PA.acquired() + US.acquired();
    const uint32_t totalSave = PA.size + US.size;

    return DAQSequence{
        .SamplesPerSec = SamplesPerSec,
        .InputRangePM_mV = InputRangePM_mv,
        .AscansPerBscan = AscansPerBscan,
        .SamplesPerAscanAcquire = totalAcquire,
        .SamplesPerAscanSave = totalSave,
        .PA = PA,
        .US = US,
    };
  }

  static constexpr DAQSequence makeDeepAcquisition202408() {
    // In this sequence, we acquire

    // Note: for the RfMeta parameters, 2730 is 1/3 of 8192 (number of samples
    // saved per Ascan), and we allocate 1/3 to PA and 2/3 to US, so some maths
    // are based on it

    constexpr auto seq = make(180e6, 800, 1000, RfMeta(2730 - 100, 250),
                              RfMeta(2730 * 2 + 2 + 100, 500));
    static_assert(seq.SamplesPerAscanSave == 8192);

    return seq;
  }

  [[nodiscard]] DepthInfo depthInfo(const float soundSpeed = 1500.0F) const {
    const auto fs = static_cast<float>(SamplesPerSec);

    const auto PA_samples2mm = soundSpeed / fs * 1000;
    const auto US_samples2mm = soundSpeed / fs * 1000 * 2;

    const auto PAdepth_mm = static_cast<float>(PA.acquired()) * PA_samples2mm;
    const auto USdepth_mm = static_cast<float>(US.acquired()) * US_samples2mm;

    const auto PAtruncate_mm = static_cast<float>(PA.truncated) * PA_samples2mm;
    const auto UStruncate_mm = static_cast<float>(US.truncated) * US_samples2mm;

    return DepthInfo{
        .PAdepth = static_cast<float>(PA.acquired()) * PA_samples2mm,
        .USdepth = static_cast<float>(US.acquired()) * US_samples2mm,
        .PAtruncated = static_cast<float>(PA.truncated) * PA_samples2mm,
        .UStruncated = static_cast<float>(US.truncated) * US_samples2mm,
    };
  }
};

constexpr uint32_t BinfileHeaderV1Magic = 0x87654321;
struct RfFileHeaderV1 {
  // Unique identifier for the header version
  uint32_t magicBits{BinfileHeaderV1Magic};

  // Size of the header in bytes
  uint32_t headerSize{sizeof(RfFileHeaderV1)};

  // DAQ input range in plus-minus millivolts
  uint32_t InputRangePM_mV{800};

  // Sampling frequency
  uint32_t SamplesPerSec{static_cast<uint32_t>(180e6)};

  // Samples per Ascan saved in the file
  uint32_t SamplesPerAscan{8192};

  // Ascans per Bscan in thie file
  uint32_t AscansPerBscan{1000};

  RfMeta PA;
  RfMeta US;

  [[nodiscard]] static RfFileHeaderV1 fromSequence(const DAQSequence &seq) {}
};

// NOLINTEND(*-magic-numbers)

} // namespace uspam
