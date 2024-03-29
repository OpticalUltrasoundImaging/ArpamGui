#include <Eigen/Dense>
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>

#include "libarpam/io.hpp"
#include "libarpam/recon.hpp"
#include <fftconv.hpp>

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
  FIRFilterParams<double> params{-1, Eigen::ArrayXd::LinSpaced(5, 0.0F, 0.4F),
                                 Eigen::ArrayXd::Ones(5)};
  ASSERT_FALSE(params.validate());
}

TEST_F(FIRFilterParamsTest, ValidatesFreqGainSize) {
  FIRFilterParams<double> params{5, Eigen::ArrayXd::LinSpaced(5, 0.0F, 0.4F),
                                 Eigen::ArrayXd::Ones(4)};
  ASSERT_FALSE(params.validate());
}

TEST_F(FIRFilterParamsTest, ValidatesFreqStartsAtZero) {
  FIRFilterParams<double> params{5, Eigen::ArrayXd::LinSpaced(5, 0.1F, 0.4F),
                                 Eigen::ArrayXd::Ones(5)};
  ASSERT_FALSE(params.validate());
}

TEST_F(FIRFilterParamsTest, PassesValidation) {
  FIRFilterParams<double> params{5, Eigen::ArrayXd::LinSpaced(5, 0.0F, 0.4F),
                                 Eigen::ArrayXd::Ones(5)};
  ASSERT_TRUE(params.validate());
}

TEST_F(FIRFilterTest, ForwardSpanCorrectSize) {
  Eigen::ArrayXd freq(4);
  Eigen::ArrayXd gain(4);
  freq << 0.0, 0.1, 0.3, 1.0;
  gain << 0.0, 1.0, 1.0, 0.0;

  FIRFilter filter(5, {15, freq, gain});

  // TODO(tnie): get values for these
  Eigen::ArrayXd input(50);
  input << 0.00000000e+00, 1.27877162e-01, 2.53654584e-01, 3.75267005e-01,
      4.90717552e-01, 5.98110530e-01, 6.95682551e-01, 7.81831482e-01,
      8.55142763e-01, 9.14412623e-01, 9.58667853e-01, 9.87181783e-01,
      9.99486216e-01, 9.95379113e-01, 9.74927912e-01, 9.38468422e-01,
      8.86599306e-01, 8.20172255e-01, 7.40277997e-01, 6.48228395e-01,
      5.45534901e-01, 4.33883739e-01, 3.15108218e-01, 1.91158629e-01,
      6.40702200e-02, -6.40702200e-02, -1.91158629e-01, -3.15108218e-01,
      -4.33883739e-01, -5.45534901e-01, -6.48228395e-01, -7.40277997e-01,
      -8.20172255e-01, -8.86599306e-01, -9.38468422e-01, -9.74927912e-01,
      -9.95379113e-01, -9.99486216e-01, -9.87181783e-01, -9.58667853e-01,
      -9.14412623e-01, -8.55142763e-01, -7.81831482e-01, -6.95682551e-01,
      -5.98110530e-01, -4.90717552e-01, -3.75267005e-01, -2.53654584e-01,
      -1.27877162e-01, -2.44929360e-16;

  Eigen::ArrayXd output = Eigen::ArrayXd::Zero(50);

  Eigen::ArrayXd expected(50);
  expected << -0.03667143, 0.04213159, 0.1417299, 0.2283456, 0.30714469,
      0.37728277, 0.43999873, 0.49482524, 0.54122433, 0.57873654, 0.60674591,
      0.62479252, 0.63258006, 0.62998065, 0.61703698, 0.59396158, 0.56113334,
      0.51909132, 0.46852582, 0.41026715, 0.3452719, 0.27460729, 0.19943364,
      0.1209853, 0.04055038, -0.04055038, -0.1209853, -0.19943364, -0.27460729,
      -0.3452719, -0.41026715, -0.46852582, -0.51909132, -0.56113334,
      -0.59396158, -0.61703698, -0.62998065, -0.63258006, -0.62479252,
      -0.60674591, -0.57873654, -0.54122433, -0.49482524, -0.43999873,
      -0.37728277, -0.30714469, -0.2283456, -0.1417299, -0.04213159, 0.03667143;

  const std::span<const double> input_span{input.data(),
                                           static_cast<size_t>(input.size())};
  const std::span<double> output_span{output.data(),
                                      static_cast<size_t>(output.size())};

  filter.forward(input_span, output_span);

  for (int i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-8);
  }
}

TEST_F(FIRFilterTest, ForwardMatrixCorrectSize) {
  FIRFilter filter(
      5, FIRFilterParams<double>{5, Eigen::ArrayXd::LinSpaced(5, 0.0, 0.4),
                                 Eigen::ArrayXd::Ones(5)});

  Eigen::MatrixXd input = Eigen::MatrixXd::Ones(10, 2); // 10x2 matrix of ones
  Eigen::MatrixXd output(10, 2);                        // Output buffer

  filter.forward(input, output);

  // Similar to the span test, we check if the output buffer has been modified
  ASSERT_NE(output, Eigen::MatrixXd::Zero(10, 2));
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