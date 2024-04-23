#include <gtest/gtest.h>

#include "uspam/imutil.hpp"
#include <armadillo>

TEST(FlipLRInplace, Correct) {
  arma::mat inp(5, 5, arma::fill::randn);
  arma::mat expected = arma::fliplr(inp);
  uspam::imutil::fliplr_inplace(inp);
  ASSERT_TRUE(arma::approx_equal(inp, expected, "absdiff", 1e-9));
}
