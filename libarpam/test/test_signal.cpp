#include <array>

#include <gtest/gtest.h>

#include "libarpam/signal.hpp"

// NOLINTBEGIN(*-magic-numbers,*-constant-array-index)

class InterpTest : public ::testing::Test {};

TEST_F(InterpTest, NormalOperation) {
  std::array x{0.5, 1.5, 2.5};
  std::array xp{0., 1., 2., 3.};
  std::array fp{0., 1., 4., 9.};
  std::array expected{0.5, 2.5, 6.5};
  std::array result{0., 0., 0.};

  arpam::signal::interp(x, xp, fp, result);
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST_F(InterpTest, AtSamplePoints) {
  std::array x{1., 2.};
  std::array xp{0., 1., 2., 3.};
  std::array fp{0., 1., 4., 9.};
  std::array expected{1., 4.}; // Exactly at sample points
  std::array result{0., 0.};

  arpam::signal::interp(x, xp, fp, result);

  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5);
  }
}

TEST_F(InterpTest, OutOfBounds) {
  std::array x{-1., 4.};
  std::array xp{0., 1., 2., 3.};
  std::array fp{0., 1., 4., 9.};

  std::array expected{0., 9.}; // Clamp to the nearest value
  std::array result{0., 0.};
  arpam::signal::interp(x, xp, fp, result);

  EXPECT_NEAR(result[0], expected[0], 1e-5);
  EXPECT_NEAR(result[1], expected[1], 1e-5);
}

TEST_F(InterpTest, ErrorConditions) {
  std::array x{0.5, 1.5, 2.5};
  std::array xp{1.}; // Invalid: not enough elements
  std::array fp{1.};
  std::array result{0., 0., 0.};

  EXPECT_THROW(arpam::signal::interp(x, xp, fp, result), std::invalid_argument);
}

TEST_F(InterpTest, MismatchedXpFpSizes) {
  std::array x{0.5, 1.5, 2.5};
  std::array xp{0., 1., 2.}; // Mismatched sizes
  std::array fp{0., 1.};
  std::array result{0., 0., 0.};

  EXPECT_THROW(arpam::signal::interp(x, xp, fp, result), std::invalid_argument);
}

class Firwin2Test : public ::testing::Test {};

TEST_F(Firwin2Test, ReturnsCorrectNumberOfTaps) {
  const int numtaps = 11;
  Eigen::ArrayXd freq(2);
  freq << 0.0, 1.0;
  Eigen::ArrayXd gain(2);
  gain << 0.0, 1.0;

  auto result = arpam::signal::firwin2(numtaps, freq, gain);
  EXPECT_EQ(result.size(), numtaps);
}

TEST_F(Firwin2Test, HandlesInvalidInputs) {
  int numtaps = -1; // Invalid number of taps
  Eigen::ArrayXd freq(2);
  freq << 0.0, 1.0;
  Eigen::ArrayXd gain(2);
  gain << 0.0, 1.0;

  EXPECT_THROW(arpam::signal::firwin2(numtaps, freq, gain),
               std::invalid_argument);
}

TEST_F(Firwin2Test, NormalOperation) {
  // Compare results to scipy.signal.firwin2
  {
    const int numtaps = 65;
    Eigen::ArrayXd freq(4);
    freq << 0., 0.1, 0.3, 1.;
    Eigen::ArrayXd gain(4);
    gain << 0., 1., 1., 0.;

    Eigen::ArrayXd expected(numtaps);
    expected << -1.55718185e-04, -1.67573816e-04, -2.33823175e-04,
        -2.32192842e-04, -2.91158967e-04, -2.51668428e-04, -3.26376966e-04,
        -2.65653718e-04, -3.71843796e-04, -2.03609636e-04, -2.41842178e-04,
        1.17433661e-04, -2.18084155e-05, 1.85547861e-04, -5.56396602e-04,
        -7.21128139e-04, -2.05692040e-03, -2.28176667e-03, -4.38698043e-03,
        -5.31680279e-03, -9.65408517e-03, -1.20243639e-02, -1.84700044e-02,
        -1.97250185e-02, -2.61864876e-02, -2.42862613e-02, -3.47521750e-02,
        -3.28942920e-02, -5.80710686e-02, -4.47551532e-02, -9.49538191e-02,
        1.79790415e-01, 5.99940709e-01, 1.79790415e-01, -9.49538191e-02,
        -4.47551532e-02, -5.80710686e-02, -3.28942920e-02, -3.47521750e-02,
        -2.42862613e-02, -2.61864876e-02, -1.97250185e-02, -1.84700044e-02,
        -1.20243639e-02, -9.65408517e-03, -5.31680279e-03, -4.38698043e-03,
        -2.28176667e-03, -2.05692040e-03, -7.21128139e-04, -5.56396602e-04,
        1.85547861e-04, -2.18084155e-05, 1.17433661e-04, -2.41842178e-04,
        -2.03609636e-04, -3.71843796e-04, -2.65653718e-04, -3.26376966e-04,
        -2.51668428e-04, -2.91158967e-04, -2.32192842e-04, -2.33823175e-04,
        -1.67573816e-04, -1.55718185e-04;

    Eigen::ArrayXd kernel = arpam::signal::firwin2(numtaps, freq, gain);

    for (int i = 0; i < kernel.size(); ++i) {
      EXPECT_NEAR(kernel[i], expected[i], 1e-5);
    }
  }

  {
    const int numtaps = 129;

    Eigen::ArrayXd freq(4);
    freq << 0.0, 0.2, 0.4, 1.0;
    Eigen::ArrayXd gain(4);
    gain << 0, 0.5, 0.5, 0;

    Eigen::ArrayXd expected(numtaps);
    expected << -1.04440356e-05, -6.51672129e-06, -7.29156588e-06,
        1.26129140e-06, -4.17179901e-07, 1.43562754e-06, -1.01615958e-05,
        -1.06130286e-05, -1.94968342e-05, -1.42101328e-05, -2.43881481e-05,
        -1.65477432e-05, -1.96420131e-05, 3.74980351e-06, -8.59236164e-07,
        4.66299586e-06, -3.14639119e-05, -3.34882713e-05, -6.17579263e-05,
        -4.50973055e-05, -7.71512631e-05, -5.22449029e-05, -6.11880570e-05,
        1.20235303e-05, -1.68501275e-06, 1.48080811e-05, -9.39550721e-05,
        -9.95476272e-05, -1.81563008e-04, -1.31568878e-04, -2.23344630e-04,
        -1.50703556e-04, -1.74789341e-04, 3.52579481e-05, -2.69534947e-06,
        4.34751858e-05, -2.65838414e-04, -2.83314115e-04, -5.17780454e-04,
        -3.77444421e-04, -6.45330681e-04, -4.40501638e-04, -5.15314444e-04,
        1.07838384e-04, -3.65205676e-06, 1.38291880e-04, -8.44806881e-04,
        -9.28226892e-04, -1.74992577e-03, -1.32458003e-03, -2.36436140e-03,
        -1.69915700e-03, -2.10317149e-03, 4.78053801e-04, -4.33204095e-06,
        7.31275612e-04, -4.94584606e-03, -6.26647007e-03, -1.40694187e-02,
        -1.33259303e-02, -3.20048197e-02, -3.48775071e-02, -8.17650824e-02,
        6.21128089e-02, 2.99995422e-01, 6.21128089e-02, -8.17650824e-02,
        -3.48775071e-02, -3.20048197e-02, -1.33259303e-02, -1.40694187e-02,
        -6.26647007e-03, -4.94584606e-03, 7.31275612e-04, -4.33204095e-06,
        4.78053801e-04, -2.10317149e-03, -1.69915700e-03, -2.36436140e-03,
        -1.32458003e-03, -1.74992577e-03, -9.28226892e-04, -8.44806881e-04,
        1.38291880e-04, -3.65205676e-06, 1.07838384e-04, -5.15314444e-04,
        -4.40501638e-04, -6.45330681e-04, -3.77444421e-04, -5.17780454e-04,
        -2.83314115e-04, -2.65838414e-04, 4.34751858e-05, -2.69534947e-06,
        3.52579481e-05, -1.74789341e-04, -1.50703556e-04, -2.23344630e-04,
        -1.31568878e-04, -1.81563008e-04, -9.95476272e-05, -9.39550721e-05,
        1.48080811e-05, -1.68501275e-06, 1.20235303e-05, -6.11880570e-05,
        -5.22449029e-05, -7.71512631e-05, -4.50973055e-05, -6.17579263e-05,
        -3.34882713e-05, -3.14639119e-05, 4.66299586e-06, -8.59236164e-07,
        3.74980351e-06, -1.96420131e-05, -1.65477432e-05, -2.43881481e-05,
        -1.42101328e-05, -1.94968342e-05, -1.06130286e-05, -1.01615958e-05,
        1.43562754e-06, -4.17179901e-07, 1.26129140e-06, -7.29156588e-06,
        -6.51672129e-06, -1.04440356e-05;

    Eigen::ArrayXd kernel = arpam::signal::firwin2(numtaps, freq, gain);

    for (int i = 0; i < kernel.size(); ++i) {
      EXPECT_NEAR(kernel[i], expected[i], 1e-5);
    }
  }
}

// NOLINTEND(*-magic-numbers,*-constant-array-index)
