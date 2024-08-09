#include "Recon.hpp"
#include "uspam/ioParams.hpp"

#include <future>

namespace Recon {

QImage cvMatToQImage(const cv::Mat &mat) {
  switch (mat.type()) {
  // 8-bit, 4 channel
  case CV_8UC4: {
    QImage image(mat.data, mat.cols, mat.rows, static_cast<qsizetype>(mat.step),
                 QImage::Format_ARGB32);
    return image.copy(); // Use copy to detach from original data
  }
  // 8-bit, 3 channel
  case CV_8UC3: {
    QImage image(mat.data, mat.cols, mat.rows, static_cast<qsizetype>(mat.step),
                 QImage::Format_RGB888);
    return image.rgbSwapped(); // Convert BGR to RGB
  }
  // 8-bit, 1 channel
  case CV_8UC1: {
    QImage image(mat.data, mat.cols, mat.rows, static_cast<qsizetype>(mat.step),
                 QImage::Format_Grayscale8);
    return image.copy();
  }
  // 64F, 1 channel
  case CV_64FC1: {
    cv::Mat mat_normalized = mat * 255; // NOLINT
    cv::Mat mat_u8;
    mat_normalized.convertTo(mat_u8, CV_8U);
    return cvMatToQImage(mat_u8);
  }

  default:
    qWarning("cvMatToQImage() - cv::Mat image type not handled in switch: %d",
             mat.type());
    break;
  }
  return {};
}

void reconBScan(BScanData<ArpamFloat> &data,
                const uspam::recon::ReconParams2 &params,
                const uspam::io::IOParams &ioparams) {
  auto &perfMetrics = data.metrics;
  uspam::TimeIt timeit;

  /*
  Here, data has defined
    data.rf
  */

  /*
  Split and background
  */
  // Estimate background from current RF
  // const arma::Col<ArpamFloat> background_aline = arma::mean(data.rf, 1);

  // Split RF into PA and US scan lines
  {
    const uspam::TimeIt timeit;
    ioparams.splitRfPAUS(data.rf, data.PA.rf, data.US.rf);
    perfMetrics.split_ms = timeit.get_ms();
  }

  /*
  Recon
  */
  const auto &paramsPA = params.PA;
  const auto &paramsUS = params.US;

  const bool flip = uspam::recon::ReconParams::flip(data.frameIdx);

  constexpr bool USE_ASYNC = true;
  if constexpr (USE_ASYNC) {
    auto a1 = std::async(std::launch::async, procOne, std::ref(paramsPA),
                         std::ref(data.PA), flip);

    auto a2 = std::async(std::launch::async, procOne, std::ref(paramsUS),
                         std::ref(data.US), flip);

    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] = a1.get();
      perfMetrics.beamform_ms = beamform_ms;
      perfMetrics.recon_ms = recon_ms;
      perfMetrics.imageConversion_ms = imageConversion_ms;
    }

    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] = a2.get();
      perfMetrics.beamform_ms += beamform_ms;
      perfMetrics.recon_ms += recon_ms;
      perfMetrics.imageConversion_ms += imageConversion_ms;
    }

  } else {

    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] =
          procOne(paramsPA, data.PA, flip);
      perfMetrics.beamform_ms = beamform_ms;
      perfMetrics.recon_ms = recon_ms;
      perfMetrics.imageConversion_ms = imageConversion_ms;
    }

    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] =
          procOne(paramsUS, data.US, flip);
      perfMetrics.beamform_ms += beamform_ms;
      perfMetrics.recon_ms += recon_ms;
      perfMetrics.imageConversion_ms += imageConversion_ms;
    }
  }

  // Compute scalebar scalar
  // fct is the depth [m] of one radial pixel
  data.fct = [&data] {
    constexpr double soundSpeed = 1500.0; // [m/s] Sound speed
    constexpr double fs = 180e6;          // [1/s] Sample frequency

    // [m] multiplier to convert sampled US points to meters. 2x travel path
    constexpr double fctRect = soundSpeed / fs / 2;

    // [points]
    const auto USpoints_rect = static_cast<double>(data.US.rf.n_rows);

    // [points]
    const auto USpoints_radial = static_cast<double>(data.US.radial.rows) / 2;

    // [m]
    const auto fctRadial = fctRect * USpoints_rect / USpoints_radial;
    return fctRadial;
  }();

  {
    const uspam::TimeIt timeit;
    uspam::imutil::makeOverlay(data.US.radial, data.PA.radial, data.PAUSradial);
    perfMetrics.overlay_ms = timeit.get_ms();
  }

  data.PAUSradial_img = cvMatToQImage(data.PAUSradial);

  perfMetrics.total_ms = timeit.get_ms();
}

void saveImages(BScanData<ArpamFloat> &data, const fs::path &saveDir) {
  // Save to file
  const uspam::TimeIt timeit;

  const int i = data.frameIdx;

  constexpr bool CONCURRENT_SAVE = true;
  if constexpr (!CONCURRENT_SAVE) {
    /*
    Sequential version
    */
    data.US.radial_img.save(
        path2QString(saveDir / std::format("US_{:03d}.png", i)));
    data.PA.radial_img.save(
        path2QString(saveDir / std::format("PA_{:03d}.png", i)));
    data.PAUSradial_img.save(
        path2QString(saveDir / std::format("PAUS_{:03d}.png", i)));
  } else {
    /*
    Concurrent version
    */
    auto *pool = QThreadPool::globalInstance();

    // using snprintf because apple clang doesn't support std::format yet...
    // NOLINTBEGIN(*-magic-numbers,*-pointer-decay,*-avoid-c-arrays)
    char _buf[64];
    std::snprintf(_buf, sizeof(_buf), "US_%03d.png", i);
    auto fname = path2QString(saveDir / std::string(_buf));
    pool->start(new ImageWriteTask(data.US.radial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PA_%03d.png", i);
    fname = path2QString(saveDir / std::string(_buf));
    pool->start(new ImageWriteTask(data.PA.radial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PAUS_%03d.png", i);
    fname = path2QString(saveDir / std::string(_buf));
    pool->start(new ImageWriteTask(data.PAUSradial_img, fname));
    // NOLINTEND(*-magic-numbers,*-pointer-decay,*-avoid-c-arrays)
  }
}

std::tuple<float, float, float> procOne(const uspam::recon::ReconParams &params,
                                        BScanData_<T> &data, bool flip) {
  auto &rf = data.rf;
  auto &rfBeamformed = data.rfBeamformed;
  auto &rfEnv = data.rfEnv;
  auto &rfLog = data.rfLog;

  rfBeamformed.set_size(rf.n_rows, rf.n_cols);
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

  // Preprocessing (flip, rotate)
  if (flip) {
    // Flip
    uspam::imutil::fliplr_inplace(rf);

    // Rotate
    if (params.rotateOffset != 0) {
      rf = arma::shift(rf, params.rotateOffset, 1);
    }
  }

  // Beamform
  {
    uspam::TimeIt timeit;
    beamform(rf, rfBeamformed, params.beamformerType, params.beamformerParams);
    beamform_ms = timeit.get_ms();
  }

  switch (params.filterType) {
  case uspam::recon::FilterType::FIR: {
    // Recon (with FIR filter)
    uspam::TimeIt timeit;

    // compute FIR filter kernels
    const auto kernel = [&] {
      std::array<double, 6> freq = {0,
                                    params.bpLowFreq,
                                    params.bpLowFreq,
                                    params.bpHighFreq,
                                    params.bpHighFreq,
                                    1};
      std::array<double, 6> gain = {0, 0, 1, 1, 0, 0};

      if constexpr (std::is_same_v<T, double>) {
        return uspam::signal::firwin2<double>(params.firTaps, freq, gain);
      } else {
        const auto _kernel =
            uspam::signal::firwin2<double>(params.firTaps, freq, gain);
        const auto kernel = arma::conv_to<arma::Col<T>>::from(_kernel);
        return kernel;
      }
    }();

    // Apply filter and envelope
    const cv::Range range(0, static_cast<int>(rf.n_cols));
    std::vector<T> _filt(rf.n_rows);
    for (int i = range.start; i < range.end; ++i) {
      const auto _rf = rfBeamformed.unsafe_col(i);
      auto _env = rfEnv.unsafe_col(i);
      fftconv::oaconvolve_fftw_same<T>(_rf, kernel, _filt);
      uspam::signal::hilbert_abs_r2c<T>(_filt, _env);
    }

    // Log compress
    constexpr float fct_mV2V = 1.0F / 1000;
    uspam::recon::logCompress<T>(rfEnv, rfLog, params.noiseFloor_mV * fct_mV2V,
                                 params.desiredDynamicRange);

    recon_ms = timeit.get_ms();
  } break;
  case uspam::recon::FilterType::IIR: {
    // Recon (with IIR filter)
    uspam::TimeIt timeit;

    // compute IIR filter kernels
    const kfr::zpk<T> filt = iir_bandpass(kfr::butterworth<T>(params.iirOrder),
                                          params.bpLowFreq, params.bpHighFreq);
    const kfr::iir_params<T> bqs = to_sos(filt);

    // Apply filter and envelope
    const cv::Range range(0, static_cast<int>(rf.n_cols));
    // cv::parallel_for_(range, [&](const cv::Range &range) {
    kfr::univector<T> _filt;
    for (int i = range.start; i < range.end; ++i) {
      const auto *const _rf = rfBeamformed.colptr(i);
      _filt = kfr::iir(kfr::make_univector(_rf, rfBeamformed.n_rows),
                       kfr::iir_params{bqs});

      auto _env = rfEnv.unsafe_col(i);
      uspam::signal::hilbert_abs_r2c<T>(_filt, _env);
    }

    // Log compress
    constexpr float fct_mV2V = 1.0F / 1000;
    uspam::recon::logCompress<T>(rfEnv, rfLog, params.noiseFloor_mV * fct_mV2V,
                                 params.desiredDynamicRange);
    // });
    recon_ms = timeit.get_ms();
  } break;
  }

  // {
  //   uspam::TimeIt timeit;
  //   uspam::recon::reconOneScan<T>(params, data.rfBeamformed, data.rfEnv,
  //                                 data.rfLog, flip);
  //   recon_ms = timeit.get_ms();
  // }

  // Truncate the pulser/laser artifact
  if (params.truncate > 0) {
    rfLog.head_rows(params.truncate - 1).zeros();
  }

  {
    uspam::TimeIt timeit;
    data.radial = uspam::imutil::makeRadial(data.rfLog);
    data.radial_img = cvMatToQImage(data.radial);
    imageConversion_ms = timeit.get_ms();
  }

  return std::tuple{beamform_ms, recon_ms, imageConversion_ms};
}

} // namespace Recon
