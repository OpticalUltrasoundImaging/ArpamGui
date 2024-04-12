#include <cstdint>
#include <gtest/gtest.h>

// Assuming swap_endian_inplace is defined in `swap_endian_inplace.h`
#include "uspam/io.hpp"

// NOLINTBEGIN(*-using-namespace,*-magic-numbers,*-reinterpret-cast,*-pointer-arithmetic)

using namespace uspam::io;

TEST(SwapEndianInPlaceTest, SwapUint16) {
  uint16_t original = 0x1234;
  uint16_t expected = 0x3412;
  swap_endian_inplace(&original);
  EXPECT_EQ(original, expected);
}

TEST(SwapEndianInPlaceTest, SwapUint32) {
  uint32_t original = 0x12345678;
  uint32_t expected = 0x78563412;
  swap_endian_inplace(&original);
  EXPECT_EQ(original, expected);
}

TEST(SwapEndianInPlaceTest, SwapUint64) {
  uint64_t original = 0x123456789ABCDEF0;
  uint64_t expected = 0xF0DEBC9A78563412;
  swap_endian_inplace(&original);
  EXPECT_EQ(original, expected);
}

TEST(SwapEndianInPlaceTest, SwapInt32) {
  int32_t original = 0x12345678;
  int32_t expected = 0x78563412;
  swap_endian_inplace(&original);
  EXPECT_EQ(original, expected);
}

TEST(SwapEndianInPlaceTest, SwapFloat) {
  float original = 1.2345F;
  float expected = 0; // Expected value needs to be set based on the
                      // endianness swap of `original`
  char *expectedBytes = reinterpret_cast<char *>(&expected);
  char *originalBytes = reinterpret_cast<char *>(&original);
  std::reverse_copy(originalBytes, originalBytes + sizeof(float),
                    expectedBytes);

  swap_endian_inplace(&original);
  EXPECT_EQ(original, expected);
}

TEST(SwapEndianInPlaceTest, SwapDouble) {
  double original = 1.2345678901234567;
  double expected = 0; // Expected value needs to be set based on the endianness
                       // swap of `original`
  char *expectedBytes = reinterpret_cast<char *>(&expected);
  char *originalBytes = reinterpret_cast<char *>(&original);
  std::reverse_copy(originalBytes, originalBytes + sizeof(double),
                    expectedBytes);

  swap_endian_inplace(&original);
  EXPECT_EQ(original, expected);
}

// NOLINTEND(*-using-namespace,*-magic-numbers,*-reinterpret-cast,*-pointer-arithmetic)
