#include "uspam/recon.hpp"

namespace uspam::recon {

void recon(const arma::mat &rf, const arma::vec &kernel, arma::mat &env) {
  // cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
  //   arma::vec rf_filt(rf.n_rows);
  //   for (int i = range.start; i < range.end; ++i) {
  //     const auto src = rf.unsafe_col(i);
  //     auto dst = env.unsafe_col(i);
  //     fftconv::oaconvolve_fftw_same<double>(src, kernel, rf_filt);
  //     signal::hilbert_abs(rf_filt, dst);
  //   }
  // });

  arma::vec rf_filt(rf.n_rows);
  for (int i = 0; i < rf.n_cols; ++i) {
    const auto src = rf.unsafe_col(i);
    auto dst = env.unsafe_col(i);
    fftconv::oaconvolve_fftw_same<double>(src, kernel, rf_filt);
    signal::hilbert_abs_r2c(rf_filt, dst);
  }
}

void ReconParams2::reconOneScan(io::PAUSpair<double> &rf,
                                io::PAUSpair<uint8_t> &rfLog, bool flip) const {

  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf.PA);
    imutil::fliplr_inplace(rf.US);

    // Do rotate
    const auto rotate_offset = this->aline_rotation_offset;
    rf.PA = arma::shift(rf.PA, rotate_offset, 1);
    rf.US = arma::shift(rf.US, rotate_offset, 1);
  }

  // compute filter kernels
  const auto kernelPA = signal::firwin2(95, filter_freq_PA, filter_gain_PA);
  const auto kernelUS = signal::firwin2(95, filter_freq_US, filter_gain_US);

  auto env = io::PAUSpair<double>::empty_like(rf);

  recon(rf.PA, kernelPA, env.PA);
  logCompress<double>(env.PA, rfLog.PA, this->noise_floor_PA,
                      this->desired_dynamic_range_PA);

  recon(rf.US, kernelUS, env.US);
  logCompress<double>(env.US, rfLog.US, this->noise_floor_US,
                      this->desired_dynamic_range_US);
}

auto ReconParams2::reconOneScan(io::PAUSpair<double> &rf, bool flip) const
    -> io::PAUSpair<uint8_t> {
  auto rfLog = io::PAUSpair<uint8_t>::zeros_like(rf);
  reconOneScan(rf, rfLog, flip);
  return rfLog;
}

void ReconParams::reconOneScan(arma::Mat<double> &rf, arma::Mat<uint8_t> &rfLog,
                               bool flip) const {
  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf);

    // Do rotate
    rf = arma::shift(rf, rotate_offset, 1);
  }

  // compute filter kernels
  const auto kernel = signal::firwin2(95, filter_freq, filter_gain);

  arma::Mat<double> env(rf.n_rows, rf.n_cols, arma::fill::none);

  recon(rf, kernel, env);
  logCompress<double>(env, rfLog, noise_floor, desired_dynamic_range);
}

} // namespace uspam::recon
