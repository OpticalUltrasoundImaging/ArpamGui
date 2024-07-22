#include "uspam/beamformer/SAFT.hpp"
#include "uspam/beamformer/beamformer.hpp"
#include "uspam/reconParams.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <variant>

// NOLINTBEGIN(*-magic-numbers, *global*)

void Expect_EQ_BeamformerParams(
    const uspam::beamformer::BeamformerParams<float> &p1,
    const uspam::beamformer::BeamformerParams<float> &p2) {
  using T = float;

  if (std::holds_alternative<uspam::beamformer::SaftDelayParams<T>>(p1)) {
    EXPECT_TRUE(
        std::holds_alternative<uspam::beamformer::SaftDelayParams<T>>(p2));

    const auto &_p1 = std::get<uspam::beamformer::SaftDelayParams<T>>(p1);
    const auto &_p2 = std::get<uspam::beamformer::SaftDelayParams<T>>(p2);

    EXPECT_EQ(_p1.rt, _p2.rt);
    EXPECT_EQ(_p1.vs, _p2.vs);
    EXPECT_EQ(_p1.dt, _p2.dt);
    EXPECT_EQ(_p1.dt, _p2.dt);
    EXPECT_EQ(_p1.da, _p2.da);

    EXPECT_EQ(_p1.f, _p2.f);
    EXPECT_EQ(_p1.d, _p2.d);

    EXPECT_EQ(_p1.illumAngleDeg, _p2.illumAngleDeg);

  } else if (std::holds_alternative<std::monostate>(p1)) {
    EXPECT_TRUE(std::holds_alternative<std::monostate>(p2));
  } else {
    throw std::runtime_error("Untested type");
  }
}

// Test for ReconParams
TEST(ReconParamsTest, CopyConstructor) {

  uspam::recon::ReconParams params1{{0, 0.03, 0.035, 0.2, 0.22, 1},
                                    {0, 0, 1, 1, 0, 0},
                                    250,
                                    25,
                                    9.0F,
                                    35.0F,
                                    uspam::beamformer::BeamformerType::SAFT_CF};
  uspam::recon::ReconParams params2 = params1; // Copy constructor

  EXPECT_EQ(params1.filterFreq, params2.filterFreq);
  EXPECT_EQ(params1.filterGain, params2.filterGain);
  EXPECT_EQ(params1.truncate, params2.truncate);
  EXPECT_EQ(params1.rotateOffset, params2.rotateOffset);
  EXPECT_EQ(params1.noiseFloor_mV, params2.noiseFloor_mV);
  EXPECT_EQ(params1.desiredDynamicRange, params2.desiredDynamicRange);
  EXPECT_EQ(params1.beamformerType, params2.beamformerType);
  Expect_EQ_BeamformerParams(params1.beamformerParams,
                             params2.beamformerParams);
}

TEST(ReconParamsTest, CopyAssignment) {
  uspam::recon::ReconParams params1{{0, 0.03, 0.035, 0.2, 0.22, 1},
                                    {0, 0, 1, 1, 0, 0},
                                    250,
                                    25,
                                    9.0F,
                                    35.0F,
                                    uspam::beamformer::BeamformerType::SAFT_CF};
  uspam::recon::ReconParams params2;
  params2 = params1; // Copy assignment

  EXPECT_EQ(params1.filterFreq, params2.filterFreq);
  EXPECT_EQ(params1.filterGain, params2.filterGain);
  EXPECT_EQ(params1.truncate, params2.truncate);
  EXPECT_EQ(params1.rotateOffset, params2.rotateOffset);
  EXPECT_EQ(params1.noiseFloor_mV, params2.noiseFloor_mV);
  EXPECT_EQ(params1.desiredDynamicRange, params2.desiredDynamicRange);
  EXPECT_EQ(params1.beamformerType, params2.beamformerType);
  Expect_EQ_BeamformerParams(params1.beamformerParams,
                             params2.beamformerParams);
}

TEST(ReconParamsTest, MoveConstructor) {
  uspam::recon::ReconParams params1{{0, 0.03, 0.035, 0.2, 0.22, 1},
                                    {0, 0, 1, 1, 0, 0},
                                    250,
                                    25,
                                    9.0F,
                                    35.0F,
                                    uspam::beamformer::BeamformerType::SAFT_CF};
  uspam::recon::ReconParams params2 = std::move(params1); // Move constructor

  EXPECT_EQ(params2.filterFreq,
            std::vector<double>({0, 0.03, 0.035, 0.2, 0.22, 1}));
  EXPECT_EQ(params2.filterGain, std::vector<double>({0, 0, 1, 1, 0, 0}));
  EXPECT_EQ(params2.truncate, 250);
  EXPECT_EQ(params2.rotateOffset, 25);
  EXPECT_EQ(params2.noiseFloor_mV, 9.0F);
  EXPECT_EQ(params2.desiredDynamicRange, 35.0F);
  EXPECT_EQ(params2.beamformerType, uspam::beamformer::BeamformerType::SAFT_CF);
  // TODO need to check the BeamformerParams
}

TEST(ReconParams2Test, MoveAssignment) {
  uspam::recon::ReconParams2 params1 =
      uspam::recon::ReconParams2::system2024v1();
  uspam::recon::ReconParams2 params2;
  params2 = std::move(params1); // Move assignment

  EXPECT_EQ(params2.PA.filterFreq,
            std::vector<double>({0, 0.03, 0.035, 0.2, 0.22, 1}));
  EXPECT_EQ(params2.PA.filterGain, std::vector<double>({0, 0, 1, 1, 0, 0}));
  EXPECT_EQ(params2.PA.truncate, 250);
  EXPECT_EQ(params2.PA.rotateOffset, 25);
  EXPECT_EQ(params2.PA.noiseFloor_mV, 9.0F);
  EXPECT_EQ(params2.PA.desiredDynamicRange, 35.0F);
  EXPECT_EQ(params2.PA.beamformerType,
            uspam::beamformer::BeamformerType::SAFT_CF);
  // TODO need to check the BeamformerParams
}

// NOLINTEND(*-magic-numbers, *global*)