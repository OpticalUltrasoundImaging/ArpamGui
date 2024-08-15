#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/beamformer/SAFT.hpp"
#include <armadillo>
#include <gtest/gtest.h>
#include <numbers>

// NOLINTBEGIN(*-magic-numbers,*-constant-array-index,*-global-variables,*-goto)

namespace beamformer = uspam::beamformer;

TEST(SaftTimeDelayParamTest, Correct) {
  beamformer::SaftDelayParams<double> saftParams{};
  saftParams.rt = 6.2;
  saftParams.vs = 1.5e3;
  saftParams.dt = 1.0 / 180e6;
  saftParams.da = 2 * std::numbers::pi / 1000;
  saftParams.f = 15.0;
  saftParams.d = 8.5;
  saftParams.illumAngleDeg = 5;

  const auto timeDelay =
      beamformer::computeSaftTimeDelay(saftParams, 769, 2450);

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
  using T = float;

  const auto saftParams = beamformer::SaftDelayParams<T>::make();
  const auto timeDelay =
      beamformer::computeSaftTimeDelay(saftParams, 769, 2450);

  using beamformer::BeamformerType;
  const arma::Mat<T> rf(2500, 1000, arma::fill::randn);

  {
    const auto rf_saft =
        beamformer::apply_saft<T, BeamformerType::SAFT>(timeDelay, rf);
    const auto rf_saft_cf =
        beamformer::apply_saft<T, BeamformerType::SAFT_CF>(timeDelay, rf);
  }
  {
    const auto rf_saft =
        beamformer::apply_saft_v2<T, BeamformerType::SAFT>(timeDelay, rf);
    const auto rf_saft_cf =
        beamformer::apply_saft_v2<T, BeamformerType::SAFT_CF>(timeDelay, rf);
  }

  // rf_saft_cf.save("rf_saft_cf.bin", arma::raw_binary);
  // TODO write tests
}

// NOLINTEND(*-magic-numbers,*-constant-array-index,*-global-variables,*-goto)