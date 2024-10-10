#include "uspam/beamformer/SAFT.hpp"
#include "uspam/beamformer/BeamformerType.hpp"

namespace uspam::beamformer {

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft(const TimeDelay<Float> &timeDelay,
                            const arma::Mat<Float> &rf) {
  const int nScans = rf.n_cols;
  const int nPts = rf.n_rows;

  arma::Mat<Float> rf_saft = rf; // copy
  arma::Mat<uint8_t> n_saft(rf.n_rows, rf.n_cols, arma::fill::ones);
  arma::Mat<Float> CF_denom = arma::square(rf);

  const auto range = cv::Range(timeDelay.zStart, timeDelay.zEnd);
#if defined(WIN32)
  cv::parallel_for_(range, [&](const cv::Range range) {
#endif
    for (int j = 0; j < nScans; ++j) {
      for (int iz = range.start; iz < range.end; ++iz) {
        // for (int iz = timeDelay.zStart; iz < timeDelay.zEnd; ++iz) {

        const auto NLines = timeDelay.saftLines.at(iz - timeDelay.zStart);

        for (int dj_saft = 0; dj_saft < NLines; ++dj_saft) {
          const auto dt =
              timeDelay.timeDelay.at(iz - timeDelay.zStart, dj_saft);
          const int iz_delayed = static_cast<int>(std::round(iz + dt));

          if (iz_delayed >= nPts) {
            continue;
          }

          const auto val = rf.at(iz_delayed, j);
          const auto valSq = val * val;

          {
            const auto j_saft = (j - dj_saft + nScans) % nScans;
            rf_saft.at(iz, j_saft) += val;
            CF_denom.at(iz, j_saft) += valSq;
            n_saft.at(iz, j_saft) += 1;
          }

          {
            const auto j_saft = (j + dj_saft + nScans) % nScans;
            rf_saft.at(iz, j_saft) += val;
            CF_denom.at(iz, j_saft) += valSq;
            n_saft.at(iz, j_saft) += 1;
          }
        }
      }
    }
#if defined(WIN32)
  });
#endif

  // CF = PA_saft ** 2 / (CF_denom * n_saft)
  // rf_saft_cf = rf_saft * CF / n_saft

  // Apply coherence factor
  for (int col = 0; col < rf_saft.n_cols; ++col) {
    for (int row = 0; row < rf_saft.n_rows; ++row) {
      const auto rf_saft_ = rf_saft.at(row, col);
      const auto n_saft_ = n_saft.at(row, col);

      if constexpr (BfType == BeamformerType::SAFT) {
        rf_saft.at(row, col) = rf_saft_ / n_saft_;
      } else { // BfType == BeamformerType::SAFT_CF
        const auto nom = rf_saft_ * rf_saft_;
        const auto denom = CF_denom.at(row, col) * n_saft_;

        const auto CF_ = denom != 0 ? nom / denom : 1;
        rf_saft.at(row, col) = rf_saft_ * CF_ / n_saft_;
      }
    }
  }

  return rf_saft;
}

template arma::Mat<float>
apply_saft<float, BeamformerType::SAFT>(const TimeDelay<float> &timeDelay,
                                        const arma::Mat<float> &rf);

template arma::Mat<float>
apply_saft<float, BeamformerType::SAFT_CF>(const TimeDelay<float> &timeDelay,
                                           const arma::Mat<float> &rf);

template <Floating Float, BeamformerType BfType>
arma::Mat<Float> apply_saft_v2(const TimeDelay<Float> &timeDelay,
                               const arma::Mat<Float> &rf,
                               const size_t truncated) {
  const int nScans = rf.n_cols;
  const int nPts = rf.n_rows;

  // arma::zeros(rf.n_rows, rf.n_cols);
  // arma::Mat<Float>::zeros();
  // Here, rf is already truncated
  arma::Mat<Float> rf_saft = rf; // copy
  arma::Mat<uint8_t> n_saft(rf.n_rows, rf.n_cols, arma::fill::ones);
  arma::Mat<Float> CF_denom = arma::square(rf);

  // const auto range = cv::Range(timeDelay.zStart, timeDelay.zEnd);
  const auto range = cv::Range(0, nScans);
  //   cv::parallel_for_(range, [&](const cv::Range range) {
  //   for (int sc = 0; sc < nScans; ++sc) {
  //     constexpr int max_saft_lines = 15;
  //     for (int ch = std::max(0, -max_saft_lines);
  //          ch < std::max(max_saft_lines, nScans); ++ch) {
  //     }
  //   }

  for (int j = range.start; j < range.end; ++j) {
    // for (int j = 0; j < nScans; ++j) {
    //   for (int iz = range.start; iz < range.end; ++iz) {
    for (int iz = timeDelay.zStart; iz < timeDelay.zEnd; ++iz) {
      const int iz_truncated = iz - truncated;
      if (iz_truncated < 0) {
        continue;
      }

      const auto NLines = timeDelay.saftLines.at(iz - timeDelay.zStart);

      for (int dj_saft = 0; dj_saft < NLines; ++dj_saft) {
        const auto dt = timeDelay.timeDelay(iz - timeDelay.zStart, dj_saft);
        const int iz_delayed =
            static_cast<int>(std::round(iz + dt)) - truncated;

        if (iz_delayed >= nPts || iz_delayed < 0) {
          continue;
        }

        const auto val = rf(iz_delayed, j);
        const auto valSq = val * val;

        {
          const auto j_saft = (j - dj_saft + nScans) % nScans;
          rf_saft(iz_truncated, j_saft) += val;
          CF_denom(iz_truncated, j_saft) += valSq;
          n_saft(iz_truncated, j_saft) += 1;
        }

        {
          const auto j_saft = (j + dj_saft + nScans) % nScans;
          rf_saft(iz_truncated, j_saft) += val;
          CF_denom(iz_truncated, j_saft) += valSq;
          n_saft(iz_truncated, j_saft) += 1;
        }
      }
    }
  }
  //   });

  // CF = PA_saft ** 2 / (CF_denom * n_saft)
  // rf_saft_cf = rf_saft * CF / n_saft

  // Apply coherence factor
  for (int col = 0; col < rf_saft.n_cols; ++col) {
    for (int row = 0; row < rf_saft.n_rows; ++row) {
      const auto rf_saft_ = rf_saft(row, col);
      auto n_saft_ = n_saft(row, col);

      if (n_saft_ <= 0) {
        // Trap here
        n_saft_ = 1;
      }

      if constexpr (BfType == BeamformerType::SAFT) {
        rf_saft.at(row, col) = rf_saft_ / n_saft_;

      } else { // BfType == BeamformerType::SAFT_CF
        const auto nom = rf_saft_ * rf_saft_;
        const auto denom = CF_denom(row, col) * n_saft_;

        const auto CF_ = denom != 0 ? nom / denom : 1;
        rf_saft.at(row, col) = rf_saft_ * CF_ / n_saft_;
      }
    }
  }

  return rf_saft;
}

template arma::Mat<float>
apply_saft_v2<float, BeamformerType::SAFT>(const TimeDelay<float> &timeDelay,
                                           const arma::Mat<float> &rf,
                                           size_t truncated);
template arma::Mat<double>
apply_saft_v2<double, BeamformerType::SAFT>(const TimeDelay<double> &timeDelay,
                                            const arma::Mat<double> &rf,
                                            size_t truncated);

template arma::Mat<float>
apply_saft_v2<float, BeamformerType::SAFT_CF>(const TimeDelay<float> &timeDelay,
                                              const arma::Mat<float> &rf,
                                              size_t truncated);
template arma::Mat<double> apply_saft_v2<double, BeamformerType::SAFT_CF>(
    const TimeDelay<double> &timeDelay, const arma::Mat<double> &rf,
    size_t truncated);

} // namespace uspam::beamformer