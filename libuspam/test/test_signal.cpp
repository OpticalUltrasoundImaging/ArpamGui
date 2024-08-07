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

  EXPECT_THROW(interp<double>(x, xp, fp, result),
               std::invalid_argument);
}

TEST(InterpTest, MismatchedXpFpSizes) {
  std::array x{0.5, 1.5, 2.5};
  std::array xp{0., 1., 2.}; // Mismatched sizes
  std::array fp{0., 1.};
  std::array result{0., 0., 0.};

  EXPECT_THROW(interp<double>(x, xp, fp, result),
               std::invalid_argument);
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

  EXPECT_THROW(firwin2<double>(numtaps, freq, gain),
               std::invalid_argument);
}

TEST(Firwin2Test, HandlesFreqGainDifferentSize) {
  const int numtaps = 5; // Invalid number of taps
  {
    const arma::vec freq({0.0, 0.1, 0.3, 1.0});
    const arma::vec gain({0.0, 1.0, 0.0});
    EXPECT_THROW(firwin2<double>(numtaps, freq, gain),
                 std::invalid_argument);
  }
  {
    const arma::vec freq({0.0, 0.1, 1.0});
    const arma::vec gain({0.0, 1.0, 1.0, 0.0});
    EXPECT_THROW(firwin2<double>(numtaps, freq, gain),
                 std::invalid_argument);
  }
}

TEST(Firwin2Test, ValidatesFreqStartsAtZero) {
  const int numtaps = 5; // Invalid number of taps
  const arma::vec freq({0.1, 0.1, 0.3, 1.0});
  const arma::vec gain({0.0, 1.0, 1.0, 0.0});
  EXPECT_THROW(firwin2<double>(numtaps, freq, gain),
               std::invalid_argument);
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

TEST(HilbertTest, CorrectEven1) {
  const arma::vec input(
      {0.70426012, 0.9910894,  0.25996975, 0.03048985, 0.44575919, 0.00234811,
       0.16894339, 0.95555906, 0.78683054, 0.9074598,  0.80980262, 0.90684473,
       0.01975208, 0.26868918, 0.63096345, 0.00949143, 0.19329107, 0.47083633,
       0.1293803,  0.40263912, 0.55569273, 0.66302628, 0.35491789, 0.90795212,
       0.06281225, 0.04934021, 0.50569647, 0.16552751, 0.54134253, 0.93703689,
       0.21840856, 0.55255146, 0.44386797, 0.84914859, 0.93753191, 0.29544402,
       0.94560284, 0.03784119, 0.21626115, 0.24381063, 0.21715903, 0.56702923,
       0.7431051,  0.17921447, 0.87611432, 0.86963978, 0.29927141, 0.17584925,
       0.06252297, 0.60827164, 0.9315844,  0.9945137,  0.7051144,  0.4224271,
       0.03583113, 0.93855678, 0.5511782,  0.88677877, 0.0803159,  0.85090798,
       0.82234698, 0.91333603, 0.46843806, 0.77417541});

  const arma::vec expected(
      {0.72285818, 1.04454556, 0.8640911,  0.034432,   0.46100612, 0.12114978,
       0.64446857, 1.03076146, 0.78705816, 0.9087384,  0.8354315,  1.07082577,
       0.66143963, 0.33112718, 0.72374387, 0.31799882, 0.29194159, 0.47579193,
       0.13453679, 0.48344689, 0.59228153, 0.66875575, 0.36331236, 0.91446944,
       0.5895158,  0.37403857, 0.51038254, 0.22833903, 0.76205164, 0.938234,
       0.26308539, 0.64184866, 0.44680141, 0.97424037, 1.05243629, 0.2964168,
       0.98289496, 0.51480138, 0.274161,   0.24606205, 0.386044,   0.67845279,
       0.74318278, 0.19197491, 1.03159005, 0.97593795, 0.34355628, 0.24584989,
       0.47891027, 0.79671491, 1.03621001, 1.01077447, 0.71491148, 0.65426397,
       0.43720109, 0.95844141, 0.5605923,  0.90859147, 0.08045817, 0.96808119,
       0.82278087, 0.92466687, 0.51351057, 0.77584509});

  // const arma::vec env = arma::abs(hilbert(input));
  // for (int i = 0; i < env.size(); ++i) {
  //   EXPECT_NEAR(env[i], expected[i], 1e-8);
  // }

  {
    const arma::vec env2 = hilbert_abs<double>(input);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }

  {
    arma::vec env2(input.size(), arma::fill::none);
    hilbert_abs<double>(input, env2);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }

  {
    arma::vec env2(input.size(), arma::fill::none);
    hilbert_abs_r2c<double>(input, env2);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }
}

TEST(HilbertTest, CorrectEven2) {
  const arma::vec input(
      {0.67303061, 0.07031093, 0.02299068, 0.03994196, 0.06774837, 0.57030347,
       0.39383879, 0.55341641, 0.7353254,  0.95355522, 0.62350182, 0.08210663,
       0.21392645, 0.50596058, 0.71389642, 0.38035662, 0.1709159,  0.8574199,
       0.12506465, 0.1448533,  0.22439503, 0.43571134, 0.59857477, 0.72014218,
       0.91910707, 0.28881895, 0.96931641, 0.87464548, 0.28261418, 0.18948964,
       0.52817823, 0.67686381, 0.19552923, 0.4541282,  0.91158641, 0.05348329,
       0.65869846, 0.40253366, 0.15745584, 0.0826146,  0.36339561, 0.27259175,
       0.20902166, 0.59232833, 0.15662596, 0.51309582, 0.87275145, 0.68777038,
       0.95574306, 0.26720851, 0.98672858, 0.31535309, 0.79193634, 0.19681808,
       0.08589511, 0.90976332, 0.30769337, 0.77479057, 0.15506799, 0.77064977,
       0.47687615, 0.1433175,  0.19568917, 0.10289787, 0.18685338, 0.63863646,
       0.25914749, 0.83672874, 0.83794641, 0.36417784, 0.54542362, 0.86962947,
       0.94524803, 0.1565067});

  const arma::vec expected(
      {0.70537986, 0.59066141, 0.04195806, 0.06847073, 0.4673644,  0.64117007,
       0.43267877, 0.63682801, 0.77284939, 0.95491222, 0.81100906, 0.27261927,
       0.28817496, 0.54812605, 0.71390609, 0.51417855, 0.30535807, 0.85779674,
       0.43513656, 0.27410295, 0.27224209, 0.65350644, 0.62657378, 0.82820738,
       0.94498675, 0.29340552, 1.00213295, 0.99657213, 0.51985405, 0.19063387,
       0.54847204, 0.69525365, 0.3230105,  0.61337015, 0.97951657, 0.21410588,
       0.668536,   0.58435699, 0.21531106, 0.08606706, 0.42218768, 0.27752384,
       0.40051056, 0.61083315, 0.17951973, 0.83083664, 0.88095995, 0.72476405,
       0.98649352, 0.26934684, 0.98736796, 0.49412493, 0.79742265, 0.7445879,
       0.58350856, 0.91931938, 0.30843144, 0.79513653, 0.1962957,  0.78014359,
       0.69925226, 0.22144526, 0.2070008,  0.10876435, 0.43948166, 0.6753348,
       0.30710702, 0.98247372, 0.88915557, 0.36442081, 0.55774301, 0.89400091,
       1.11166012, 0.35360697});

  // const arma::vec env = arma::abs(hilbert(input));
  // for (int i = 0; i < env.size(); ++i) {
  //   EXPECT_NEAR(env[i], expected[i], 1e-8);
  // }

  {
    const arma::vec env2 = hilbert_abs<double>(input);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }

  {
    arma::vec env2(input.size(), arma::fill::none);
    hilbert_abs<double>(input, env2);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }

  {
    arma::vec env2(input.size(), arma::fill::none);
    hilbert_abs_r2c<double>(input, env2);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }
}

TEST(HilbertTest, CorrectOdd1) {
  const arma::vec input(
      {0.67303061, 0.07031093, 0.02299068, 0.03994196, 0.06774837, 0.57030347,
       0.39383879, 0.55341641, 0.7353254,  0.95355522, 0.62350182, 0.08210663,
       0.21392645, 0.50596058, 0.71389642, 0.38035662, 0.1709159,  0.8574199,
       0.12506465, 0.1448533,  0.22439503, 0.43571134, 0.59857477, 0.72014218,
       0.91910707, 0.28881895, 0.96931641, 0.87464548, 0.28261418, 0.18948964,
       0.52817823, 0.67686381, 0.19552923, 0.4541282,  0.91158641, 0.05348329,
       0.65869846, 0.40253366, 0.15745584, 0.0826146,  0.36339561, 0.27259175,
       0.20902166, 0.59232833, 0.15662596, 0.51309582, 0.87275145, 0.68777038,
       0.95574306, 0.26720851, 0.98672858, 0.31535309, 0.79193634, 0.19681808,
       0.08589511, 0.90976332, 0.30769337, 0.77479057, 0.15506799, 0.77064977,
       0.47687615, 0.1433175,  0.19568917, 0.10289787, 0.18685338, 0.63863646,
       0.25914749, 0.83672874, 0.83794641, 0.36417784, 0.54542362, 0.86962947,
       0.94524803, 0.1565067,  0.04305163});

  const arma::vec expected(
      {0.68857553, 0.48206971, 0.06357346, 0.03994225, 0.48369734, 0.65793679,
       0.43821232, 0.6493063,  0.77624788, 0.95411529, 0.80507689, 0.25930018,
       0.293755,   0.55244065, 0.71398309, 0.50839731, 0.31099803, 0.85762273,
       0.42911317, 0.27852107, 0.27563785, 0.65634698, 0.62826393, 0.8294748,
       0.94373882, 0.29365754, 1.00348814, 0.99641361, 0.5155005,  0.19055782,
       0.54988257, 0.69564845, 0.31892896, 0.61154463, 0.97763269, 0.21771487,
       0.66945349, 0.58780564, 0.21169422, 0.08462572, 0.42501176, 0.27889751,
       0.4053606,  0.60892169, 0.1824702,  0.82358087, 0.88181871, 0.72150015,
       0.9849121,  0.27111984, 0.98763772, 0.50468761, 0.79830965, 0.75949614,
       0.59127484, 0.92200829, 0.30913992, 0.79988621, 0.19065418, 0.77688086,
       0.69152022, 0.24269817, 0.20333174, 0.10294056, 0.45281796, 0.66348087,
       0.31743238, 0.95652835, 0.88101748, 0.37418033, 0.56726639, 0.87369254,
       1.07268045, 0.71883469, 0.10876629});

  // const arma::vec env = arma::abs(hilbert(input));
  // for (int i = 0; i < env.size(); ++i) {
  //   EXPECT_NEAR(env[i], expected[i], 1e-6);
  // }

  {
    const arma::vec env2 = hilbert_abs<double>(input);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }

  {
    arma::vec env2(input.size(), arma::fill::none);
    hilbert_abs<double>(input, env2);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }

  {
    arma::vec env2(input.size(), arma::fill::none);
    hilbert_abs_r2c<double>(input, env2);
    for (int i = 0; i < env2.size(); ++i) {
      EXPECT_NEAR(env2[i], expected[i], 1.5e-8);
    }
  }
}

// NOLINTEND(*-numbers,*-constant-array-index,*-global-variables,*-goto)
