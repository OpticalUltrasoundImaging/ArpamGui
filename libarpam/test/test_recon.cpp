#include <Eigen/Dense>
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>

#include "libarpam/io.hpp"
#include "libarpam/recon.hpp"

// NOLINTBEGIN(*-magic-numbers,*-constant-array-index)

class FIRFilterTest : public ::testing::Test {
protected:
  // You can define common setup logic here if necessary
  void SetUp() override {
    // Common setup for each test, if needed
  }
};

class FIRFilterParamsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Save cerr's buffer
    sbuf = std::cerr.rdbuf();
    // Redirect cerr to our stringstream buffer
    std::cerr.rdbuf(buffer.rdbuf());
  }
  void TearDown() override {
    std::cerr.rdbuf(sbuf);
    sbuf = nullptr;
  }

private:
  std::stringstream buffer{};
  std::streambuf *sbuf{};
};

using arpam::recon::FIRFilterParams;
using arpam::recon::pipeline::FIRFilter;

TEST_F(FIRFilterParamsTest, ValidatesNumtaps) {
  FIRFilterParams<float> params{-1, Eigen::ArrayXf::LinSpaced(5, 0.0F, 0.4F),
                                Eigen::ArrayXf::Ones(5)};
  ASSERT_FALSE(params.validate());
}

TEST_F(FIRFilterParamsTest, ValidatesFreqGainSize) {
  FIRFilterParams<float> params{5, Eigen::ArrayXf::LinSpaced(5, 0.0F, 0.4F),
                                Eigen::ArrayXf::Ones(4)};
  ASSERT_FALSE(params.validate());
}

TEST_F(FIRFilterParamsTest, ValidatesFreqStartsAtZero) {
  FIRFilterParams<float> params{5, Eigen::ArrayXf::LinSpaced(5, 0.1F, 0.4F),
                                Eigen::ArrayXf::Ones(5)};
  ASSERT_FALSE(params.validate());
}

TEST_F(FIRFilterParamsTest, PassesValidation) {
  FIRFilterParams<float> params{5, Eigen::ArrayXf::LinSpaced(5, 0.0F, 0.4F),
                                Eigen::ArrayXf::Ones(5)};
  ASSERT_TRUE(params.validate());
}

TEST_F(FIRFilterTest, ForwardSpanCorrectSize) {
  FIRFilterParams<float> params{5, Eigen::ArrayXf::LinSpaced(5, 0.0F, 0.4F),
                                Eigen::ArrayXf::Ones(5)};
  FIRFilter<float> filter(5, params);

  // TODO(tnie): get values for these
  std::vector<float> input(10, 1.0F);
  std::vector<float> output(10, 0.0F);

  filter.forward(input, output);

  // This test might need to be adjusted based on what you expect the output to
  // be For now, let's just check if the output buffer has been modified
  // (assuming the filter does not produce all zeros)
  ASSERT_NE(output, std::vector<float>(10, 0.0F));
}

TEST_F(FIRFilterTest, ForwardMatrixCorrectSize) {
  FIRFilterParams<float> params{5, Eigen::ArrayXf::LinSpaced(5, 0.0F, 0.4F),
                                Eigen::ArrayXf::Ones(5)};
  FIRFilter<float> filter(5, params);

  Eigen::MatrixXf input = Eigen::MatrixXf::Ones(10, 2); // 10x2 matrix of ones
  Eigen::MatrixXf output(10, 2);                        // Output buffer

  filter.forward(input, output);

  // Similar to the span test, we check if the output buffer has been modified
  ASSERT_NE(output, Eigen::MatrixXf::Zero(10, 2));
}

class ReconTest : public ::testing::Test {
public:
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
  const int numtaps = 65;
  Eigen::ArrayXd freq(4);
  freq << 0., 0.1, 0.3, 1.;
  Eigen::ArrayXd gain(4);
  gain << 0., 1., 1., 0.;
  arpam::recon::ReconParams params{
      arpam::recon::FIRFilterParams{numtaps, freq, gain}, 45.0F};

  Eigen::MatrixX<double> result(m_rf.rows(), m_rf.cols());
  arpam::recon::recon(params, m_rf, result);
}

// NOLINTEND(*-magic-numbers,*-constant-array-index)