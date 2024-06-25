#include <armadillo>
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "uspam/reconParams.hpp"

namespace fs = std::filesystem;

// NOLINTBEGIN(*-magic-numbers,*-constant-array-index)

// class ReconTest : public ::testing::Test {
// public:
//   void SetUp() override {
//     std::filesystem::path filepath(
//         "/Users/tnie/Downloads/135245/135245/NormalUS4.bin");
//
//     if (!std::filesystem::exists(filepath)) {
//       std::cerr << "File doesn't exist: " << filepath << "\n";
//       return;
//     }
//
//     m_rf = arpam::io::load_bin<double>(filepath, std::endian::big);
//   }
//
//   Eigen::MatrixXd m_rf;
// };
//
// TEST_F(ReconTest, Basic) {
//   const int numtaps = 65;
//   Eigen::ArrayXd freq(4);
//   freq << 0., 0.1, 0.3, 1.;
//   Eigen::ArrayXd gain(4);
//   gain << 0., 1., 1., 0.;
//   arpam::recon::ReconParams params{
//       arpam::recon::FIRFilterParams{numtaps, freq, gain}, 45.0F};
//
//   Eigen::MatrixX<double> result(m_rf.rows(), m_rf.cols());
//   arpam::recon::recon(params, m_rf, result);
// }
//
//  NOLINTEND(*-magic-numbers,*-constant-array-index)

TEST(ReconParams2Serialize, ToString) {
  const auto params_true = uspam::recon::ReconParams2::system2024v1();
  {
    const auto doc = params_true.serializeToDoc();

    uspam::recon::ReconParams2 params{};

    ASSERT_NE(params_true.PA.filterFreq, params.PA.filterFreq);
    ASSERT_NE(params_true.PA.filterGain, params.PA.filterGain);
    ASSERT_NE(params_true.PA.noiseFloor_mV, params.PA.noiseFloor_mV);
    ASSERT_NE(params_true.PA.desiredDynamicRange,
              params.PA.desiredDynamicRange);
    ASSERT_NE(params_true.PA.rotateOffset, params.PA.rotateOffset);

    ASSERT_NE(params_true.US.filterFreq, params.US.filterFreq);
    ASSERT_NE(params_true.US.filterGain, params.US.filterGain);
    ASSERT_NE(params_true.US.noiseFloor_mV, params.US.noiseFloor_mV);
    ASSERT_NE(params_true.US.desiredDynamicRange,
              params.US.desiredDynamicRange);
    ASSERT_NE(params_true.US.rotateOffset, params.US.rotateOffset);

    params.deserialize(doc);

    ASSERT_EQ(params_true.PA.filterFreq, params.PA.filterFreq);
    ASSERT_EQ(params_true.PA.filterGain, params.PA.filterGain);
    ASSERT_EQ(params_true.PA.noiseFloor_mV, params.PA.noiseFloor_mV);
    ASSERT_EQ(params_true.PA.desiredDynamicRange,
              params.PA.desiredDynamicRange);
    ASSERT_EQ(params_true.PA.rotateOffset, params.PA.rotateOffset);

    ASSERT_EQ(params_true.US.filterFreq, params.US.filterFreq);
    ASSERT_EQ(params_true.US.filterGain, params.US.filterGain);
    ASSERT_EQ(params_true.US.noiseFloor_mV, params.US.noiseFloor_mV);
    ASSERT_EQ(params_true.US.desiredDynamicRange,
              params.US.desiredDynamicRange);
    ASSERT_EQ(params_true.US.rotateOffset, params.US.rotateOffset);
  }
}

TEST(ReconParams2Serialize, ToFile) {
  const auto params_true = uspam::recon::ReconParams2::system2024v1();
  fs::path jsonFile = "tmp.json";
  {
    ASSERT_TRUE(params_true.serializeToFile(jsonFile));

    uspam::recon::ReconParams2 params{};

    ASSERT_NE(params_true.PA.filterFreq, params.PA.filterFreq);
    ASSERT_NE(params_true.PA.filterGain, params.PA.filterGain);
    ASSERT_NE(params_true.PA.noiseFloor_mV, params.PA.noiseFloor_mV);
    ASSERT_NE(params_true.PA.desiredDynamicRange,
              params.PA.desiredDynamicRange);
    ASSERT_NE(params_true.PA.rotateOffset, params.PA.rotateOffset);

    ASSERT_NE(params_true.US.filterFreq, params.US.filterFreq);
    ASSERT_NE(params_true.US.filterGain, params.US.filterGain);
    ASSERT_NE(params_true.US.noiseFloor_mV, params.US.noiseFloor_mV);
    ASSERT_NE(params_true.US.desiredDynamicRange,
              params.US.desiredDynamicRange);
    ASSERT_NE(params_true.US.rotateOffset, params.US.rotateOffset);

    ASSERT_TRUE(params.deserializeFromFile(jsonFile));

    ASSERT_EQ(params_true.PA.filterFreq, params.PA.filterFreq);
    ASSERT_EQ(params_true.PA.filterGain, params.PA.filterGain);
    ASSERT_EQ(params_true.PA.noiseFloor_mV, params.PA.noiseFloor_mV);
    ASSERT_EQ(params_true.PA.desiredDynamicRange,
              params.PA.desiredDynamicRange);
    ASSERT_EQ(params_true.PA.rotateOffset, params.PA.rotateOffset);

    ASSERT_EQ(params_true.US.filterFreq, params.US.filterFreq);
    ASSERT_EQ(params_true.US.filterGain, params.US.filterGain);
    ASSERT_EQ(params_true.US.noiseFloor_mV, params.US.noiseFloor_mV);
    ASSERT_EQ(params_true.US.desiredDynamicRange,
              params.US.desiredDynamicRange);
    ASSERT_EQ(params_true.US.rotateOffset, params.US.rotateOffset);
  }
  fs::remove(jsonFile);
}
