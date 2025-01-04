#include <array>

#include <gtest/gtest.h>

#include "uspam/signal.hpp"

// NOLINTBEGIN(*-numbers,*-constant-array-index,*-global-variables,*-goto)

using namespace uspam::signal; // NOLINT(*-using-namespace)

TEST(InterpTest, NormalOperation) {
  std::array x{0.5, 1.5, 2.5};
  std::array xp{0., 1., 2., 3.};
  std::array fp{0., 1., 4., 9.};
  std::array expected{0.5, 2.5, 6.5};
  std::array result{0., 0., 0.};

  interp<double>(x, xp, fp, result);
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(InterpTest, AtSamplePoints) {
  std::array x{1., 2.};
  std::array xp{0., 1., 2., 3.};
  std::array fp{0., 1., 4., 9.};
  std::array expected{1., 4.}; // Exactly at sample points
  std::array result{0., 0.};

  interp<double>(x, xp, fp, result);

  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5);
  }
}

TEST(InterpTest, OutOfBounds) {
  std::array x{-1., 4.};
  std::array xp{0., 1., 2., 3.};
  std::array fp{0., 1., 4., 9.};

  std::array expected{0., 9.}; // Clamp to the nearest value
  std::array result{0., 0.};
  interp<double>(x, xp, fp, result);

  EXPECT_NEAR(result[0], expected[0], 1e-5);
  EXPECT_NEAR(result[1], expected[1], 1e-5);
}

TEST(InterpTest, ErrorConditions) {
  std::array x{0.5, 1.5, 2.5};
  std::array xp{1.}; // Invalid: not enough elements
  std::array fp{1.};
  std::array result{0., 0., 0.};

  EXPECT_THROW(interp<double>(x, xp, fp, result), std::invalid_argument);
}

TEST(InterpTest, MismatchedXpFpSizes) {
  std::array x{0.5, 1.5, 2.5};
  std::array xp{0., 1., 2.}; // Mismatched sizes
  std::array fp{0., 1.};
  std::array result{0., 0., 0.};

  EXPECT_THROW(interp<double>(x, xp, fp, result), std::invalid_argument);
}

TEST(Firwin2Test, ReturnsCorrectNumberOfTaps) {
  const int numtaps = 11;
  const arma::vec freq({0.0, 1.0});
  const arma::vec gain({0.0, 1.0});

  const auto result = firwin2<double>(numtaps, freq, gain);
  EXPECT_EQ(result.size(), numtaps);
}

TEST(Firwin2Test, HandlesInvalidNumtaps) {
  const int numtaps = -1; // Invalid number of taps
  const arma::vec freq({0.0, 1.0});
  const arma::vec gain({0.0, 1.0});

  EXPECT_THROW(firwin2<double>(numtaps, freq, gain), std::invalid_argument);
}

TEST(Firwin2Test, HandlesFreqGainDifferentSize) {
  const int numtaps = 5; // Invalid number of taps
  {
    const arma::vec freq({0.0, 0.1, 0.3, 1.0});
    const arma::vec gain({0.0, 1.0, 0.0});
    EXPECT_THROW(firwin2<double>(numtaps, freq, gain), std::invalid_argument);
  }
  {
    const arma::vec freq({0.0, 0.1, 1.0});
    const arma::vec gain({0.0, 1.0, 1.0, 0.0});
    EXPECT_THROW(firwin2<double>(numtaps, freq, gain), std::invalid_argument);
  }
}

TEST(Firwin2Test, ValidatesFreqStartsAtZero) {
  const int numtaps = 5; // Invalid number of taps
  const arma::vec freq({0.1, 0.1, 0.3, 1.0});
  const arma::vec gain({0.0, 1.0, 1.0, 0.0});
  EXPECT_THROW(firwin2<double>(numtaps, freq, gain), std::invalid_argument);
}

TEST(Firwin2Test, NormalOperation) {
  // Compare results to scipy.signal.firwin2
  {
    const int numtaps = 65;
    const arma::vec freq({0., 0.1, 0.3, 1.});
    const arma::vec gain({0., 1., 1., 0.});

    const arma::vec expected(
        {-1.55718185e-04, -1.67573816e-04, -2.33823175e-04, -2.32192842e-04,
         -2.91158967e-04, -2.51668428e-04, -3.26376966e-04, -2.65653718e-04,
         -3.71843796e-04, -2.03609636e-04, -2.41842178e-04, 1.17433661e-04,
         -2.18084155e-05, 1.85547861e-04,  -5.56396602e-04, -7.21128139e-04,
         -2.05692040e-03, -2.28176667e-03, -4.38698043e-03, -5.31680279e-03,
         -9.65408517e-03, -1.20243639e-02, -1.84700044e-02, -1.97250185e-02,
         -2.61864876e-02, -2.42862613e-02, -3.47521750e-02, -3.28942920e-02,
         -5.80710686e-02, -4.47551532e-02, -9.49538191e-02, 1.79790415e-01,
         5.99940709e-01,  1.79790415e-01,  -9.49538191e-02, -4.47551532e-02,
         -5.80710686e-02, -3.28942920e-02, -3.47521750e-02, -2.42862613e-02,
         -2.61864876e-02, -1.97250185e-02, -1.84700044e-02, -1.20243639e-02,
         -9.65408517e-03, -5.31680279e-03, -4.38698043e-03, -2.28176667e-03,
         -2.05692040e-03, -7.21128139e-04, -5.56396602e-04, 1.85547861e-04,
         -2.18084155e-05, 1.17433661e-04,  -2.41842178e-04, -2.03609636e-04,
         -3.71843796e-04, -2.65653718e-04, -3.26376966e-04, -2.51668428e-04,
         -2.91158967e-04, -2.32192842e-04, -2.33823175e-04, -1.67573816e-04,
         -1.55718185e-04});

    const auto kernel = firwin2<double>(numtaps, freq, gain);
    for (int i = 0; i < kernel.size(); ++i) {
      EXPECT_NEAR(kernel[i], expected[i], 1e-8);
    }
  }

  {
    const int numtaps = 129;
    const arma::vec freq({0.0, 0.2, 0.4, 1.0});
    const arma::vec gain({0, 0.5, 0.5, 0});

    const arma::vec expected(
        {-1.04440356e-05, -6.51672129e-06, -7.29156588e-06, 1.26129140e-06,
         -4.17179901e-07, 1.43562754e-06,  -1.01615958e-05, -1.06130286e-05,
         -1.94968342e-05, -1.42101328e-05, -2.43881481e-05, -1.65477432e-05,
         -1.96420131e-05, 3.74980351e-06,  -8.59236164e-07, 4.66299586e-06,
         -3.14639119e-05, -3.34882713e-05, -6.17579263e-05, -4.50973055e-05,
         -7.71512631e-05, -5.22449029e-05, -6.11880570e-05, 1.20235303e-05,
         -1.68501275e-06, 1.48080811e-05,  -9.39550721e-05, -9.95476272e-05,
         -1.81563008e-04, -1.31568878e-04, -2.23344630e-04, -1.50703556e-04,
         -1.74789341e-04, 3.52579481e-05,  -2.69534947e-06, 4.34751858e-05,
         -2.65838414e-04, -2.83314115e-04, -5.17780454e-04, -3.77444421e-04,
         -6.45330681e-04, -4.40501638e-04, -5.15314444e-04, 1.07838384e-04,
         -3.65205676e-06, 1.38291880e-04,  -8.44806881e-04, -9.28226892e-04,
         -1.74992577e-03, -1.32458003e-03, -2.36436140e-03, -1.69915700e-03,
         -2.10317149e-03, 4.78053801e-04,  -4.33204095e-06, 7.31275612e-04,
         -4.94584606e-03, -6.26647007e-03, -1.40694187e-02, -1.33259303e-02,
         -3.20048197e-02, -3.48775071e-02, -8.17650824e-02, 6.21128089e-02,
         2.99995422e-01,  6.21128089e-02,  -8.17650824e-02, -3.48775071e-02,
         -3.20048197e-02, -1.33259303e-02, -1.40694187e-02, -6.26647007e-03,
         -4.94584606e-03, 7.31275612e-04,  -4.33204095e-06, 4.78053801e-04,
         -2.10317149e-03, -1.69915700e-03, -2.36436140e-03, -1.32458003e-03,
         -1.74992577e-03, -9.28226892e-04, -8.44806881e-04, 1.38291880e-04,
         -3.65205676e-06, 1.07838384e-04,  -5.15314444e-04, -4.40501638e-04,
         -6.45330681e-04, -3.77444421e-04, -5.17780454e-04, -2.83314115e-04,
         -2.65838414e-04, 4.34751858e-05,  -2.69534947e-06, 3.52579481e-05,
         -1.74789341e-04, -1.50703556e-04, -2.23344630e-04, -1.31568878e-04,
         -1.81563008e-04, -9.95476272e-05, -9.39550721e-05, 1.48080811e-05,
         -1.68501275e-06, 1.20235303e-05,  -6.11880570e-05, -5.22449029e-05,
         -7.71512631e-05, -4.50973055e-05, -6.17579263e-05, -3.34882713e-05,
         -3.14639119e-05, 4.66299586e-06,  -8.59236164e-07, 3.74980351e-06,
         -1.96420131e-05, -1.65477432e-05, -2.43881481e-05, -1.42101328e-05,
         -1.94968342e-05, -1.06130286e-05, -1.01615958e-05, 1.43562754e-06,
         -4.17179901e-07, 1.26129140e-06,  -7.29156588e-06, -6.51672129e-06,
         -1.04440356e-05});

    const auto kernel = firwin2<double>(numtaps, freq, gain);
    for (int i = 0; i < kernel.size(); ++i) {
      EXPECT_NEAR(kernel[i], expected[i], 1e-8);
    }
  }

  {
    const int numtaps = 11;
    const arma::vec freq({0.0, 0.1, 0.3, 1.0});
    const arma::vec gain({0, 1, 1, 0});
    const arma::vec expected({-0.00303209, -0.0108505, -0.01969895, -0.06874682,
                              0.1605826, 0.59486607, 0.1605826, -0.06874682,
                              -0.01969895, -0.0108505, -0.00303209});

    const auto kernel = firwin2<double>(numtaps, freq, gain);
    for (int i = 0; i < kernel.size(); ++i) {
      EXPECT_NEAR(kernel[i], expected[i], 1e-8);
    }
  }

  {
    const int numtaps = 15;
    const arma::vec freq({0.0, 0.1, 0.3, 1.0});
    const arma::vec gain({0, 1, 1, 0});
    const arma::vec expected({-0.0023648, -0.00519774, -0.00959635, -0.02829048,
                              -0.03180531, -0.08332539, 0.16802909, 0.59486607,
                              0.16802909, -0.08332539, -0.03180531, -0.02829048,
                              -0.00959635, -0.00519774, -0.0023648});

    const auto kernel = firwin2<double>(numtaps, freq, gain);
    for (int i = 0; i < kernel.size(); ++i) {
      EXPECT_NEAR(kernel[i], expected[i], 1e-8);
    }
  }
}

TEST(Firwin2Test, NormalOperationWithFs) {
  // Compare results to scipy.signal.firwin2
  {
    const int numtaps = 65;
    const double fs = 180e6;
    const arma::vec freq({0., 8e6, 20e6, 90e6});
    const arma::vec gain({0., 1., 1., 0.});

    const arma::vec expected(
        {-1.90497517e-04, -1.69269278e-04, -1.92241910e-04, -1.54020301e-04,
         -1.73201223e-04, -1.04951515e-04, -1.36431992e-04, -4.61744616e-05,
         -1.27599219e-04, -1.93519965e-05, -1.79451249e-04, -4.80416988e-05,
         -3.71672910e-04, -3.34964462e-04, -1.12956266e-03, -1.51825209e-03,
         -3.24647944e-03, -4.30004035e-03, -7.13746952e-03, -8.54140572e-03,
         -1.22120186e-02, -1.33066829e-02, -1.78669076e-02, -1.85285448e-02,
         -2.53545972e-02, -2.58806791e-02, -3.73227676e-02, -3.53817505e-02,
         -5.39180723e-02, -3.40443681e-02, -6.96760335e-02, 1.85419165e-01,
         5.66576277e-01,  1.85419165e-01,  -6.96760335e-02, -3.40443681e-02,
         -5.39180723e-02, -3.53817505e-02, -3.73227676e-02, -2.58806791e-02,
         -2.53545972e-02, -1.85285448e-02, -1.78669076e-02, -1.33066829e-02,
         -1.22120186e-02, -8.54140572e-03, -7.13746952e-03, -4.30004035e-03,
         -3.24647944e-03, -1.51825209e-03, -1.12956266e-03, -3.34964462e-04,
         -3.71672910e-04, -4.80416988e-05, -1.79451249e-04, -1.93519965e-05,
         -1.27599219e-04, -4.61744616e-05, -1.36431992e-04, -1.04951515e-04,
         -1.73201223e-04, -1.54020301e-04, -1.92241910e-04, -1.69269278e-04,
         -1.90497517e-04});

    const auto kernel = firwin2<double>(numtaps, freq, gain, 0, fs);

    for (int i = 0; i < kernel.size(); ++i) {
      EXPECT_NEAR(kernel[i], expected[i], 1e-8);
    }
  }
}

// NOLINTEND(*-numbers,*-constant-array-index,*-global-variables,*-goto)
