#pragma once

#include "Common.hpp"
#include "strConvUtils.hpp"
#include "uspam/recon.hpp"
#include "uspam/reconParams.hpp"
#include <QImage>
#include <QThreadPool>
#include <QtDebug>
#include <QtLogging>
#include <RFBuffer.hpp>
#include <armadillo>
#include <future>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <uspam/fft.hpp>
#include <uspam/imutil.hpp>
#include <uspam/timeit.hpp>
#include <uspam/uspam.hpp>
#include <utility>

namespace Recon {
namespace fs = std::filesystem;

struct PerformanceMetrics {
  float total_ms{};
  float fileloader_ms{};
  float splitRf_ms{};
  float beamform_ms{};
  float recon_ms{};
  float imageConversion_ms{};
  float makeOverlay_ms{};

  // Template function to handle the common formatting
  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const PerformanceMetrics &pm) {
    stream << "fileloader " << static_cast<int>(pm.fileloader_ms)
           << ", splitRf " << static_cast<int>(pm.splitRf_ms) << ", beamform "
           << static_cast<int>(pm.beamform_ms) << ", recon "
           << static_cast<int>(pm.recon_ms) << ", imageConversion "
           << static_cast<int>(pm.imageConversion_ms) << ", makeOverlay "
           << static_cast<int>(pm.makeOverlay_ms);
    return stream;
  }
};

/**
Supports CV_8UC4, CV_8UC3, CV_8UC1, CV_64FC1
Create copies for all.
 */
QImage cvMatToQImage(const cv::Mat &mat);

template <uspam::Floating T>
auto procOne(const uspam::recon::ReconParams &params, BScanData_<T> &data,
             bool flip) {
  auto &rf = data.rf;
  auto &rfBeamformed = data.rfBeamformed;
  auto &rfFilt = data.rfFilt;
  auto &rfEnv = data.rfEnv;
  auto &rfLog = data.rfLog;

  rfBeamformed.set_size(rf.n_rows, rf.n_cols);
  rfFilt.set_size(rf.n_rows, rf.n_cols);
  rfEnv.set_size(rf.n_rows, rf.n_cols);
  rfLog.set_size(rf.n_rows, rf.n_cols);

  float beamform_ms{};
  float recon_ms{};
  float imageConversion_ms{};

  /*
  Flip
  Beamform
  Filter
  Envelop detect
  Log compress
  */

  // Preprocessing (flip, rotate, truncate)
  {
    if (flip) {
      // Flip
      uspam::imutil::fliplr_inplace(rf);

      // Rotate
      rf = arma::shift(rf, params.rotateOffset, 1);
    }

    // Truncate the pulser/laser artifact
    rf.head_rows(params.truncate - 1).zeros();
  }

  // Beamform
  {
    uspam::TimeIt timeit;
    beamform(rf, rfBeamformed, params.beamformerType, params.beamformerParams);
    beamform_ms = timeit.get_ms();
  }

  // Recon
  {
    uspam::TimeIt timeit;

    // compute filter kernels
    const auto kernel = [&] {
      constexpr int numtaps = 95;
      if constexpr (std::is_same_v<T, double>) {
        return uspam::signal::firwin2<double>(numtaps, params.filterFreq,
                                              params.filterGain);
      } else {
        const auto _kernel = uspam::signal::firwin2<double>(
            numtaps, params.filterFreq, params.filterGain);
        const auto kernel = arma::conv_to<arma::Col<T>>::from(_kernel);
        return kernel;
      }
    }();

    // Apply filter and envelope
    const cv::Range range(0, static_cast<int>(rf.n_cols));
    for (int i = range.start; i < range.end; ++i) {
      const auto _rf = rfBeamformed.unsafe_col(i);
      auto _filt = rfFilt.unsafe_col(i);
      auto _env = rfEnv.unsafe_col(i);
      fftconv::oaconvolve_fftw_same<T>(_rf, kernel, _filt);
      uspam::signal::hilbert_abs_r2c<T>(_filt, _env);
    }

    // Log compress
    constexpr float fct_mV2V = 1.0F / 1000;
    uspam::recon::logCompress<T>(rfEnv, rfLog, params.noiseFloor_mV * fct_mV2V,
                                 params.desiredDynamicRange);

    recon_ms = timeit.get_ms();
  }

  // {
  //   uspam::TimeIt timeit;
  //   uspam::recon::reconOneScan<T>(params, data.rfBeamformed, data.rfEnv,
  //                                 data.rfLog, flip);
  //   recon_ms = timeit.get_ms();
  // }

  {
    uspam::TimeIt timeit;
    data.radial = uspam::imutil::makeRadial(data.rfLog);
    data.radial_img = cvMatToQImage(data.radial);
    imageConversion_ms = timeit.get_ms();
  }

  return std::tuple{beamform_ms, recon_ms, imageConversion_ms};
}

class ImageWriteTask : public QRunnable {
  QImage img;
  QString fname;

public:
  ImageWriteTask(QImage img, QString fname)
      : img(std::move(img)), fname(std::move(fname)) {}
  void run() override { img.save(fname); }
};

PerformanceMetrics reconBScan(BScanData<ArpamFloat> &data,
                              const uspam::recon::ReconParams2 &params);

void saveImages(BScanData<ArpamFloat> &data, const fs::path &saveDir);

class Reconstructor {
public:
  explicit Reconstructor(uspam::recon::ReconParams2 params =
                             uspam::recon::ReconParams2::system2024v1())
      : m_params(std::move(params)) {}

  auto recon(BScanData<ArpamFloat> &data) const {
    std::unique_lock<std::mutex> lock(m_mtx);
    auto metrics = reconBScan(data, m_params);
    return metrics;
  }

  void setParams(uspam::recon::ReconParams2 params) {
    std::unique_lock<std::mutex> lock(m_mtx);
    m_params = std::move(params);
  }

private:
  uspam::recon::ReconParams2 m_params;
  mutable std::mutex m_mtx;
};

} // namespace Recon