#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/reconParams.hpp"
#include <filesystem>
#include <gtest/gtest.h>

// NOLINTBEGIN(*-magic-numbers, *global*)

namespace fs = std::filesystem;

using uspam::beamformer::BeamformerType;
using uspam::recon::ReconParams;
using uspam::recon::ReconParams2;

#define ASSERT_PARAMS(assert_func, a, b)                                       \
  assert_func((a).bpLowFreq, (b).bpLowFreq);                                   \
  assert_func((a).bpHighFreq, (b).bpHighFreq);                                 \
  assert_func((a).noiseFloor_mV, (b).noiseFloor_mV);                           \
  assert_func((a).desiredDynamicRange, (b).desiredDynamicRange);

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
  }
  fs::remove(jsonFile);
}

// Test for ReconParams
TEST(ReconParamsTest, CopyConstructor) {
  const auto params = ReconParams2::system2024v2GUI();
  const auto &params1 = params.PA;
  ReconParams params2 = params1; // Copy constructor

  ASSERT_PARAMS(EXPECT_EQ, params1, params2);
}

TEST(ReconParamsTest, CopyAssignment) {
  const auto params = ReconParams2::system2024v2GUI();
  const auto &params1 = params.PA;

  ReconParams params2;
  params2 = params1; // Copy assignment

  EXPECT_EQ(params1.beamformerType, params2.beamformerType);
}

// NOLINTEND(*-magic-numbers, *global*)