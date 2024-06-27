#include "uspam/SAFT.hpp"
#include "uspam/signal.hpp"
#include <armadillo>
#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

namespace saft = uspam::saft;

TEST(SaftTimeDelayParamTest, Correct) {
  saft::SaftDelayParams saftParams;
  saftParams.rt = 6.2;
  saftParams.vs = 1.5e3;
  saftParams.dt = 1.0 / 180e6;
  saftParams.da = 2 * std::numbers::pi / 1000;
  saftParams.f = 15.0;
  saftParams.d = 8.5;
  saftParams.angle = std::asin(8.5 / (2 * 15.0));
  saftParams.angleLight = saft::deg2rad(5);

  const auto timeDelay = saftParams.computeSaftTimeDelay(769, 2450);

  arma::mat gt(timeDelay.timeDelay.n_rows, timeDelay.timeDelay.n_cols,
               arma::fill::none);

  gt.row(100) = arma::vec({0., -0.08699811, -0.34794024, -0.78266997,
                           -1.39092706, -2.1723482, -3.12646802, -4.25272036,
                           0., 0., 0., 0., 0., 0., 0.})
                    .t();

  ASSERT_TRUE(arma::approx_equal(timeDelay.timeDelay.row(100), gt.row(100),
                                 "absdiff", 1e-6));
}

TEST(SaftApply, Correct) {
  const auto saftParams = saft::SaftDelayParams::make();
  const auto timeDelay = saftParams.computeSaftTimeDelay(769, 2450);

  const arma::mat rf(2500, 1000, arma::fill::randn);
  const auto [rf_saft, rf_saft_cf] =
      saft::apply_saft<double, double>(timeDelay, rf);
  // rf_saft_cf.save("rf_saft_cf.bin", arma::raw_binary);
  // TODO write tests
}