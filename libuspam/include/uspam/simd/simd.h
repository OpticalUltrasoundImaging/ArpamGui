#pragma once

#include <concepts>
#include <cstdint>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/highway.h>
#include <hwy/print-inl.h>
#include <type_traits>

HWY_BEFORE_NAMESPACE();
namespace hwy::HWY_NAMESPACE {

// Log compress to range of 0 - 1
template <typename T>
void logCompress(const T *x, uint8_t *xLog, size_t size, const T noiseFloor,
                 const T desiredDynamicRangeDB = 45.0)
  requires(std::is_floating_point_v<T>)
{
  constexpr HWY_FULL(T) d;
  constexpr size_t L = Lanes(d);
  const CappedTag<int32_t, L> d_i32;
  const CappedTag<uint8_t, L> d_u8;

  // Apply log compression with clipping in a single pass
  const auto vNoiseFloorInv = Set(d, 1 / noiseFloor);
  const auto vDynamicRangeDB = Set(d, desiredDynamicRangeDB);
  const auto vDynamicRangeDBInv = Set(d, 1 / desiredDynamicRangeDB);

  for (int i = 0; i + L <= size; i += L) {
    const auto val = Load(d, x + i);

    // NOLINTNEXTLINE(*-magic-numbers)
    auto compressed = Set(d, 20.0) * Log10(d, val * vNoiseFloorInv);
    compressed =
        Clamp(compressed, Set(d, 0), vDynamicRangeDB) * vDynamicRangeDBInv;

    // NOLINTNEXTLINE(*-magic-numbers)
    constexpr T fct = 255.;
    compressed *= Set(d, fct);

    // Store
    {
      if constexpr (std::is_same_v<T, float>) {
        // float  -> i32 -> u8
        const auto v_i32 = ConvertTo(d_i32, compressed);
        const auto v_u8 = DemoteTo(d_u8, v_i32);
        Store(v_u8, d_u8, xLog + i);
      } else { // double
        // double -> i32 -> u8
        const auto v_i32 = DemoteTo(ScalableTag<int32_t>(), compressed);
        const auto v_u8 = DemoteTo(ScalableTag<uint8_t>(), v_i32);
        Store(v_u8, d_u8, xLog + i);
      }
    }
  }
}

} // namespace hwy::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();