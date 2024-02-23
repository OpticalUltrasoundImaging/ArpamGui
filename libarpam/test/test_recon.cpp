#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "libarpam/libarpam.hpp"

class ReconTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::filesystem::path filepath(
        "/Users/tnie/Downloads/135245/135245/NormalUS4.bin");

    if (!std::filesystem::exists(filepath)) {
      std::cerr << "File doesn't exist: " << filepath << "\n";
      return;
    }

    m_rf = arpam::io::load_bin<double>(filepath.c_str(), std::endian::big);
  }

  Eigen::MatrixXd m_rf;
};

TEST_F(ReconTest, Basic) {
  // for (int i = 0; i < rf.cols(); i++) {
  // Print the first column
  int i = 0;
  const auto col = m_rf.col(i).head(10);
  std::cout << col << "\n";
  // }

  Eigen::MatrixX<double> result(m_rf.rows(), m_rf.cols());
  std::cout << "recon...\n";
  arpam::recon::recon(m_rf, result);

  std::cout << "Result: " << result.col(0).head(10);
}