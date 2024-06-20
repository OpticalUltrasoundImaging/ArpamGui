#include "uspam/recon.hpp"
#include "uspam/json.hpp"
#include "uspam/signal.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace uspam::recon {

void recon(const arma::mat &rf, const arma::vec &kernel, arma::mat &env) {
  const cv::Range range(0, static_cast<int>(rf.n_cols));
  // cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
  arma::vec rf_filt(rf.n_rows);
  for (int i = range.start; i < range.end; ++i) {
    const auto src = rf.unsafe_col(i);
    auto dst = env.unsafe_col(i);
    fftconv::oaconvolve_fftw_same<double>(src, kernel, rf_filt);
    signal::hilbert_abs_r2c(rf_filt, dst);
  }
  // });
}

void reconOneScan(const ReconParams2 &params, io::PAUSpair<double> &rf,
                  io::PAUSpair<uint8_t> &rfLog, bool flip) {

  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf.PA);
    imutil::fliplr_inplace(rf.US);

    // Do rotate
    const auto rotate_offset = params.US.rotateOffset;
    rf.PA = arma::shift(rf.PA, rotate_offset, 1);
    rf.US = arma::shift(rf.US, rotate_offset, 1);
  }

  // compute filter kernels
  const auto kernelPA =
      signal::firwin2(95, params.PA.filterFreq, params.PA.filterGain);
  const auto kernelUS =
      signal::firwin2(95, params.US.filterFreq, params.US.filterGain);

  auto env = io::PAUSpair<double>::empty_like(rf);

  recon(rf.PA, kernelPA, env.PA);
  logCompress<double>(env.PA, rfLog.PA, params.PA.noiseFloor,
                      params.PA.desiredDynamicRange);

  recon(rf.US, kernelUS, env.US);
  logCompress<double>(env.US, rfLog.US, params.US.noiseFloor,
                      params.US.desiredDynamicRange);
}

auto reconOneScan(const ReconParams2 &params, io::PAUSpair<double> &rf,
                  bool flip) -> io::PAUSpair<uint8_t> {
  auto rfLog = io::PAUSpair<uint8_t>::zeros_like(rf);
  reconOneScan(params, rf, rfLog, flip);
  return rfLog;
}

void reconOneScan(const ReconParams &params, arma::Mat<double> &rf,
                  arma::Mat<uint8_t> &rfLog, bool flip) {
  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf);

    // Do rotate
    rf = arma::shift(rf, params.rotateOffset, 1);
  }

  // compute filter kernels
  const auto kernel = signal::firwin2(95, params.filterFreq, params.filterGain);

  arma::Mat<double> env(rf.n_rows, rf.n_cols, arma::fill::none);

  recon(rf, kernel, env);
  logCompress<double>(env, rfLog, params.noiseFloor,
                      params.desiredDynamicRange);
}

} // namespace uspam::recon
