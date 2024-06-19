#include <armadillo>
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

#include "uspam/io.hpp"
#include "uspam/recon.hpp"
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

    ASSERT_NE(params_true.filterFreqPA, params.filterFreqPA);
    ASSERT_NE(params_true.filterFreqUS, params.filterFreqUS);
    ASSERT_NE(params_true.filterGainPA, params.filterGainPA);
    ASSERT_NE(params_true.filterGainUS, params.filterGainUS);
    ASSERT_NE(params_true.noiseFloorPA, params.noiseFloorPA);
    ASSERT_NE(params_true.desiredDynamicRangePA, params.desiredDynamicRangePA);
    ASSERT_NE(params_true.noiseFloorUS, params.noiseFloorUS);
    ASSERT_NE(params_true.desiredDynamicRangeUS, params.desiredDynamicRangeUS);
    ASSERT_NE(params_true.alineRotationOffset, params.alineRotationOffset);

    params.deserialize(doc);
    ASSERT_EQ(params_true.filterFreqPA, params.filterFreqPA);
    ASSERT_EQ(params_true.filterFreqUS, params.filterFreqUS);
    ASSERT_EQ(params_true.filterGainPA, params.filterGainPA);
    ASSERT_EQ(params_true.filterGainUS, params.filterGainUS);
    ASSERT_EQ(params_true.noiseFloorPA, params.noiseFloorPA);
    ASSERT_EQ(params_true.desiredDynamicRangePA, params.desiredDynamicRangePA);
    ASSERT_EQ(params_true.noiseFloorUS, params.noiseFloorUS);
    ASSERT_EQ(params_true.desiredDynamicRangeUS, params.desiredDynamicRangeUS);
    ASSERT_EQ(params_true.alineRotationOffset, params.alineRotationOffset);
  }
}

TEST(ReconParams2Serialize, ToFile) {
  const auto params_true = uspam::recon::ReconParams2::system2024v1();
  fs::path jsonFile = "tmp.json";
  {
    ASSERT_TRUE(params_true.serializeToFile(jsonFile));

    uspam::recon::ReconParams2 params{};

    ASSERT_NE(params_true.filterFreqPA, params.filterFreqPA);
    ASSERT_NE(params_true.filterFreqUS, params.filterFreqUS);
    ASSERT_NE(params_true.filterGainPA, params.filterGainPA);
    ASSERT_NE(params_true.filterGainUS, params.filterGainUS);
    ASSERT_NE(params_true.noiseFloorPA, params.noiseFloorPA);
    ASSERT_NE(params_true.desiredDynamicRangePA, params.desiredDynamicRangePA);
    ASSERT_NE(params_true.noiseFloorUS, params.noiseFloorUS);
    ASSERT_NE(params_true.desiredDynamicRangeUS, params.desiredDynamicRangeUS);
    ASSERT_NE(params_true.alineRotationOffset, params.alineRotationOffset);

    ASSERT_TRUE(params.deserializeFromFile(jsonFile));
    ASSERT_EQ(params_true.filterFreqPA, params.filterFreqPA);
    ASSERT_EQ(params_true.filterFreqUS, params.filterFreqUS);
    ASSERT_EQ(params_true.filterGainPA, params.filterGainPA);
    ASSERT_EQ(params_true.filterGainUS, params.filterGainUS);
    ASSERT_EQ(params_true.noiseFloorPA, params.noiseFloorPA);
    ASSERT_EQ(params_true.desiredDynamicRangePA, params.desiredDynamicRangePA);
    ASSERT_EQ(params_true.noiseFloorUS, params.noiseFloorUS);
    ASSERT_EQ(params_true.desiredDynamicRangeUS, params.desiredDynamicRangeUS);
    ASSERT_EQ(params_true.alineRotationOffset, params.alineRotationOffset);
  }
  fs::remove(jsonFile);
}