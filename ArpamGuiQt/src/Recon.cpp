#include "Recon.hpp"
#include "Common.hpp"
#include "uspam/imutil.hpp"
#include "uspam/ioParams.hpp"
#include <future>
#include <opencv2/imgproc.hpp>

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

std::tuple<float, float, float> procOne(BScanData_<T> &data,
                                        const uspam::recon::ReconParams &params,
                                        const uspam::io::IOParams_ &ioparams,
                                        const bool flip) {
  /*
  Flip
  Beamform
  Filter
  Envelop detect
  Log compress
  */

  // Truncate the pulser/laser artifact
  const size_t truncate = std::max(params.truncate, 100);
  const int truncateClearLater = std::max(params.truncate - 100, 0);
  // Just remove `truncate` rows from RF.
  arma::Mat<T> rf =
      data.rf.submat(truncate, 0, data.rf.n_rows - 1, data.rf.n_cols - 1);
  // auto &rf = data.rf;
  const size_t N = rf.n_rows;

  auto &rfBeamformed = data.rfBeamformed;
  auto &rfEnv = data.rfEnv;
  auto &rfLog = data.rfLog;

  // rfBeamformed.set_size(N, rf.n_cols);
  // rfBeamformed.zeros();

  rfEnv.set_size(N, rf.n_cols);
  // rfEnv.zeros();

  rfLog.set_size(data.rf.n_rows, rf.n_cols);
  rfLog.zeros();

  float beamform_ms{};
  float recon_ms{};
  float imageConversion_ms{};

  // Preprocessing (flip, rotate)
  if (flip) {
    // Flip
    uspam::imutil::fliplr_inplace(rf);

    // Rotate
    if (params.rotateOffset != 0) {
      rf = arma::shift(rf, params.rotateOffset, 1);
    }
  }

  // Medfilt
  if (params.medfiltKsize > 1) {
    // NOLINTBEGIN
    cv::Mat cv_mat(rf.n_cols, rf.n_rows, uspam::imutil::getCvType<T>(),
                   (void *)rf.memptr());
    // NOLINTEND

    int ksize = params.medfiltKsize;
    ksize = ksize % 2 == 0 ? ksize + 1 : ksize; // Ensure odd
    cv::medianBlur(cv_mat, cv_mat, ksize);

    rf = arma::Mat<T>(cv_mat.ptr<T>(), rf.n_rows, rf.n_cols, true);

    medianBlur(rf, params.medfiltKsize);
  }

  // Beamform
  {
    uspam::TimeIt timeit;
    beamform(rf, rfBeamformed, params.beamformerType, params.beamformerParams,
             params.truncate + ioparams.delay);

    beamform_ms = timeit.get_ms();
  }

  switch (params.filterType) {
  case uspam::recon::FilterType::FIR: {
    // Recon (with FIR filter)
    uspam::TimeIt timeit;

    // compute FIR filter kernels
    const auto kernel = [&] {
      // NOLINTBEGIN(*-magic-numbers)
      // std::array<double, 6> freq = {0, params.bpLowFreq, params.bpHighFreq,
      // 1}; std::array<double, 6> gain = {0, 1, 1, 0};

      std::array<double, 6> freq = {0,
                                    params.bpLowFreq,
                                    params.bpLowFreq,
                                    params.bpHighFreq,
                                    params.bpHighFreq,
                                    1};
      std::array<double, 6> gain = {0, 0, 1, 1, 0, 0};
      // NOLINTEND(*-magic-numbers)

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
    cv::parallel_for_(range, [&](const cv::Range &range) {
      // NOLINTBEGIN(*-pointer-arithmetic)
      std::vector<T> _filt(N);
      for (int i = range.start; i < range.end; ++i) {

        // Filter
        // const auto _rf = rfBeamformed.unsafe_col(i);
        const auto _rf = std::span{rfBeamformed.colptr(i), N};
        fftconv::oaconvolve_fftw_same<T>(_rf, kernel, _filt);

        // Envelope
        auto _env = std::span{rfEnv.colptr(i), N};
        uspam::signal::hilbert_abs_r2c<T>(_filt, _env);

        // Log compress
        auto _rfLog = std::span{rfLog.colptr(i) + truncate, N};
        constexpr float fct_mV2V = 1.0F / 1000;
        uspam::recon::logCompress<ArpamFloat, uint8_t>(
            _env, _rfLog, params.noiseFloor_mV * fct_mV2V,
            params.desiredDynamicRange);
      }
      // NOLINTEND(*-pointer-arithmetic)
    });

    // // Log compress
    // constexpr float fct_mV2V = 1.0F / 1000;
    // uspam::recon::logCompress_par(rfEnv, rfLog, params.noiseFloor_mV *
    // fct_mV2V,
    //                               params.desiredDynamicRange);

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
    cv::parallel_for_(range, [&](const cv::Range &range) {
      kfr::univector<T> _filt;
      const auto N = rf.n_rows;
      for (int i = range.start; i < range.end; ++i) {
        // NOLINTBEGIN(*-pointer-arithmetic)
        // Filter
        const auto *const _rf = rfBeamformed.colptr(i);
        _filt = kfr::iir(kfr::make_univector(_rf, rfBeamformed.n_rows),
                         kfr::iir_params{bqs});

        // Envelope
        auto _env = std::span{rfEnv.colptr(i), N};
        uspam::signal::hilbert_abs_r2c<T>(_filt, _env);

        // Log compress
        auto _rfLog = std::span{rfLog.colptr(i) + truncate, N};
        constexpr float fct_mV2V = 1.0F / 1000;
        uspam::recon::logCompress<ArpamFloat, uint8_t>(
            _env, _rfLog, params.noiseFloor_mV * fct_mV2V,
            params.desiredDynamicRange);
        // NOLINTEND(*-pointer-arithmetic)
      }
    });

    // Log compress
    // constexpr float fct_mV2V = 1.0F / 1000;
    // uspam::recon::logCompress_par<T>(rfEnv, rfLog, params.noiseFloor_mV *
    // fct_mV2V,
    //                              params.desiredDynamicRange);
    recon_ms = timeit.get_ms();
  } break;
  }

  // {
  //   uspam::TimeIt timeit;
  //   uspam::recon::reconOneScan<T>(params, data.rfBeamformed, data.rfEnv,
  //                                 data.rfLog, flip);
  //   recon_ms = timeit.get_ms();
  // }

  if (truncateClearLater > 5) {
    rfLog.rows(params.truncate, params.truncate + truncateClearLater - 1)
        .zeros();
  }

  /*
  Post processing
  */

  {
    // Scan conversion
    uspam::TimeIt timeit;
    data.radial = uspam::imutil::makeRadial_v2(data.rfLog,
                                               params.padding + ioparams.delay);

    // cv::medianBlur(data.radial, data.radial, 3);

    data.radial_img = cvMatToQImage(data.radial);
    imageConversion_ms = timeit.get_ms();
  }

  return std::tuple{beamform_ms, recon_ms, imageConversion_ms};
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
  const bool flip = params.PA.flip(data.frameIdx);

  constexpr bool USE_ASYNC = true;
  if constexpr (USE_ASYNC) {
    auto a2 = std::async(std::launch::async, procOne, std::ref(data.US),
                         std::ref(params.US), std::ref(ioparams.US), flip);

    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] =
          procOne(data.PA, params.PA, ioparams.PA, flip);
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
          procOne(data.PA, params.PA, ioparams.PA, flip);
      perfMetrics.beamform_ms = beamform_ms;
      perfMetrics.recon_ms = recon_ms;
      perfMetrics.imageConversion_ms = imageConversion_ms;
    }

    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] =
          procOne(data.US, params.US, ioparams.PA, flip);
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

} // namespace Recon
