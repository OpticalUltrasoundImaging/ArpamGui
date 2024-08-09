#include <armadillo>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "uspam/reconParams.hpp"

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
