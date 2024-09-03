#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/beamformer/SAFT.hpp"
#include "uspam/beamformer/beamformer.hpp"
#include "uspam/reconParams.hpp"
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <variant>

// NOLINTBEGIN(*-magic-numbers, *global*)

namespace fs = std::filesystem;

using uspam::beamformer::BeamformerType;
using uspam::recon::ReconParams;
using uspam::recon::ReconParams2;

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

#define ASSERT_PARAMS(assert_func, a, b)                                       \
  assert_func((a).bpLowFreq, (b).bpLowFreq);                                   \
  assert_func((a).bpHighFreq, (b).bpHighFreq);                                 \
  assert_func((a).noiseFloor_mV, (b).noiseFloor_mV);                           \
  assert_func((a).desiredDynamicRange, (b).desiredDynamicRange);               \
  assert_func((a).rotateOffset, (b).rotateOffset);

TEST(ReconParams2Serialize, ToString) {
  const auto params_true = uspam::recon::ReconParams2::system2024v1();
  {
    const auto doc = params_true.serializeToDoc();

    uspam::recon::ReconParams2 params{};

    ASSERT_PARAMS(ASSERT_NE, params_true.PA, params.PA);
    ASSERT_NE(params_true.PA.beamformerType, params.PA.beamformerType);
    ASSERT_PARAMS(ASSERT_NE, params_true.US, params.US);
    // US default beamformerType is None (0)
    // ASSERT_NE(params_true.US.beamformerType, params.US.beamformerType);

    params.deserialize(doc);

    ASSERT_PARAMS(ASSERT_EQ, params_true.PA, params.PA);
    ASSERT_EQ(params_true.PA.beamformerType, params.PA.beamformerType);
    ASSERT_PARAMS(ASSERT_EQ, params_true.US, params.US);
    ASSERT_EQ(params_true.US.beamformerType, params.US.beamformerType);
    Expect_EQ_BeamformerParams(params_true.US.beamformerParams,
                               params.US.beamformerParams);
  }
}

TEST(ReconParams2Serialize, ToFile) {
  const auto params_true = uspam::recon::ReconParams2::system2024v1();
  fs::path jsonFile = "tmp.json";
  {
    ASSERT_TRUE(params_true.serializeToFile(jsonFile));

    uspam::recon::ReconParams2 params{};

    ASSERT_PARAMS(ASSERT_NE, params_true.PA, params.PA);
    ASSERT_PARAMS(ASSERT_NE, params_true.US, params.US);

    ASSERT_TRUE(params.deserializeFromFile(jsonFile));

    ASSERT_PARAMS(ASSERT_EQ, params_true.PA, params.PA);
    ASSERT_EQ(params_true.PA.beamformerType, params.PA.beamformerType);
    ASSERT_PARAMS(ASSERT_EQ, params_true.US, params.US);
    ASSERT_EQ(params_true.US.beamformerType, params.US.beamformerType);
    Expect_EQ_BeamformerParams(params_true.US.beamformerParams,
                               params.US.beamformerParams);
  }
  fs::remove(jsonFile);
}

// Test for ReconParams
TEST(ReconParamsTest, CopyConstructor) {
  const auto params = ReconParams2::system2024v2probe2();
  const auto &params1 = params.PA;
  ReconParams params2 = params1; // Copy constructor

  ASSERT_PARAMS(EXPECT_EQ, params1, params2);

  Expect_EQ_BeamformerParams(params1.beamformerParams,
                             params2.beamformerParams);
}

TEST(ReconParamsTest, CopyAssignment) {
  const auto params = ReconParams2::system2024v2probe2();
  const auto &params1 = params.PA;

  ReconParams params2;
  params2 = params1; // Copy assignment

  EXPECT_EQ(params1.beamformerType, params2.beamformerType);
  Expect_EQ_BeamformerParams(params1.beamformerParams,
                             params2.beamformerParams);
}

// NOLINTEND(*-magic-numbers, *global*)
