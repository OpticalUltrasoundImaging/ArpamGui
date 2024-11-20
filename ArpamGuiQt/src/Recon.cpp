#include "Recon.hpp"
#include "Common.hpp"
#include "uspam/SystemParams.hpp"
#include "uspam/beamformer/Beamformer.hpp"
#include "uspam/filter.hpp"
#include "uspam/ioParams.hpp"
#include "uspam/recon.hpp"
#include "uspam/reconParams.hpp"
#include "uspam/timeit.hpp"
#include <future>
#include <opencv2/imgproc.hpp>

namespace Recon {

// Convert a cv mat to a QImage
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
  const arma::Col<ArpamFloat> background_aline = arma::mean(data.rf, 1);

  // Split RF into PA and US scan lines
  perfMetrics.split_ms = uspam::measureTime([&] {
    ioparams.splitRfPAUS(data.rf, data.PA.rf, data.US.rf, background_aline,
                         params.PA.backgroundSubtract,
                         params.US.backgroundSubtract);
  });

  /*
  Recon
  */
  {
    const bool flip = params.system.flip(data.frameIdx);
    auto a2 = std::async(std::launch::async, procOne, std::ref(params.system),
                         std::ref(params.US), std::ref(data.US), flip, false);
    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] =
          procOne(params.system, params.PA, data.PA, flip, true);
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
  }

  // Compute scalebar scalar [mm]
  {
    data.spacingRectUS = params.system.dr() / 2;
    const auto USpoints_rect = static_cast<double>(data.US.rf.n_rows);
    const auto USpoints_radial = static_cast<double>(data.US.radial.rows) / 2;
    data.spacingRadialUS = data.spacingRectUS * USpoints_rect / USpoints_radial;
  }

  perfMetrics.overlay_ms = uspam::measureTime([&] {
    uspam::imutil::makeOverlay(data.US.radial, data.PA.radial, data.PAUSradial);
    data.PAUSradial_img = cvMatToQImage(data.PAUSradial);
  });

  perfMetrics.total_ms = timeit.get_ms();
}

std::tuple<float, float, float> procOne(const uspam::SystemParams &system,
                                        const uspam::recon::ReconParams &params,
                                        BScanData_<T> &data, bool flip,
                                        bool isPA) {
  /*
  Flip
  Beamform
  Filter
  Envelop detect
  Log compress
  */

  // Truncate the pulser/laser artifact
  const size_t truncate = params.truncate;
  const size_t N = data.rf.n_rows - truncate;

  // Preprocess RF data
  arma::Mat<T> rf = uspam::recon::preprocessRF(data.rf, truncate, flip, system);

  auto &rfBeamformed = data.rfBeamformed;
  auto &rfEnv = data.rfEnv;
  auto &rfLog = data.rfLog;

  rfEnv.set_size(N, rf.n_cols);
  rfLog.zeros(data.rf.n_rows, rf.n_cols);

  // Beamform
  const float beamform_ms = uspam::measureTime([&] {
    uspam::beamformer::beamform(rf, rfBeamformed, params.beamformerType, system,
                                params.truncate);
  });

  // Denoise
  if (params.medFiltSize > 1) {
    uspam::applyMedianFilter(rfBeamformed, params.medFiltSize);
  }

  // Recon
  const auto recon_ms = uspam::measureTime([&] {
    uspam::recon::filterAndEnvelope<T>(rfBeamformed, rfEnv, rfLog, N, truncate,
                                       params);
  });

  /*
  Post processing
  */
  const float imageConversion_ms = uspam::measureTime([&] {
    int offset =
        static_cast<int>(std::round(system.transducerOffset / system.dr()));
    if (!isPA) { // US
      offset *= 2;
    }
    data.radial = uspam::imutil::makeRadial_v3(data.rfLog, offset);

    // cv::medianBlur(data.radial, data.radial, 3);

    data.radial_img = cvMatToQImage(data.radial);
  });

  return std::tuple{beamform_ms, recon_ms, imageConversion_ms};
}

} // namespace Recon
