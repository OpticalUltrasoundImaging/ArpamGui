#include "uspam/surface.hpp"
#include "gmock/gmock.h"
#include <armadillo>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

TEST(SurfaceTest, FixSurfaceIdxZerosCase1) {
  using uspam::surface::fixSurfaceIdxMissing;

  // Test Case 1
  arma::vec idx = {0.0, 3.0, 2.0, 1.0, 2.0};
  const arma::vec expect = {2.5, 3.0, 2.0, 1.0, 2.0};

  fixSurfaceIdxMissing(idx);
  ASSERT_THAT(idx, testing::ElementsAreArray(expect));
}

TEST(SurfaceTest, FixSurfaceIdxZerosCase2) {
  using uspam::surface::fixSurfaceIdxMissing;

  // Test Case 2
  arma::vec idx = {1.0, 0.0, 0.0, 0.0, 2.0};
  const arma::vec expect = {1.0, 1.25, 1.5, 1.75, 2.0};

  fixSurfaceIdxMissing(idx);
  ASSERT_THAT(idx, testing::ElementsAreArray(expect));
}

TEST(SurfaceTest, FixSurfaceIdxZerosCase3) {
  using uspam::surface::fixSurfaceIdxMissing;

  // Test Case 3
  arma::vec idx = {2.0, 3.0, 2.0, 1.0, 0.0};
  const arma::vec expect = {2.0, 3.0, 2.0, 1.0, 1.5};

  fixSurfaceIdxMissing(idx);
  ASSERT_THAT(idx, testing::ElementsAreArray(expect));
}

TEST(SurfaceTest, FixSurfaceIdxZerosCaseComplex) {
  using uspam::surface::fixSurfaceIdxMissing;

  // Test complex
  arma::vec idx = {0.0, 4.0, 0.0, 0.0, 1.0, 0.0};
  const arma::vec expect = {3.0, 4.0, 3.0, 2.0, 1.0, 2.0};

  fixSurfaceIdxMissing(idx);
  ASSERT_THAT(idx, testing::ElementsAreArray(expect));
}
