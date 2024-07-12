#include "DataProcWorker.hpp"
#include "strConvUtils.hpp"
#include "uspam/recon.hpp"
#include <QTextStream>
#include <QThreadPool>
#include <QtDebug>
#include <QtLogging>
#include <armadillo>
#include <cstdio>
#include <future>
#include <sstream>
#include <tuple>
#include <uspam/fft.hpp>
#include <uspam/imutil.hpp>
#include <uspam/timeit.hpp>
#include <uspam/uspam.hpp>
#include <utility>

namespace {

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

} // namespace

void DataProcWorker::initDataBuffers() {
  QMutexLocker lock(&m_paramsMutex);
  m_data = std::make_shared<BScanData<FloatType>>();

  m_data->frameIdx = m_frameIdx;
}

void DataProcWorker::setBinfile(const fs::path &binfile) {
  m_binfilePath = binfile;
  m_imageSaveDir = m_binfilePath.parent_path() / m_binfilePath.stem();

  if (!fs::create_directory(m_imageSaveDir) && !fs::exists(m_imageSaveDir)) {
    emit error(tr("Failed to create imageSaveDir ") +
               path2QString(m_imageSaveDir));
  } else {
    emit error(tr("Saving images to ") + path2QString(m_imageSaveDir));
  }

  try {
    // Init loader
    m_loader.setParams(m_ioparams);
    m_loader.open(m_binfilePath);
    emit maxFramesChanged(m_loader.size());

    // Save init params
    saveParamsToFile();

    // Process the first frame
    playOne(0);

    m_ready = true;

  } catch (const std::runtime_error &e) {
    const auto msg = QString("DataProcWorker exception: ") +
                     QString::fromStdString(e.what());
    qWarning() << msg;
    emit error(msg);
  }
}

void DataProcWorker::play() {
  m_isPlaying = true;

  while (m_isPlaying && m_frameIdx < m_loader.size()) {
    playOne(m_frameIdx);
    m_frameIdx++;
  }

  if (m_isPlaying) {
    emit error("DataProcWorker::play Finished.");
  } else {
    emit error("DataProcWorker::play Paused.");
  }
  m_isPlaying = false;

  emit finishedPlaying();
}

void DataProcWorker::playOne(int idx) {
  m_frameIdx = idx;

  processCurrentFrame();
}
void DataProcWorker::replayOne() { processCurrentFrame(); }

void DataProcWorker::pause() { m_isPlaying = false; }

void DataProcWorker::updateParams(uspam::recon::ReconParams2 params,
                                  uspam::io::IOParams ioparams) {
  QMutexLocker lock(&m_paramsMutex);
  this->m_params = std::move(params);
  this->m_ioparams = ioparams;
}

void DataProcWorker::saveParamsToFile() {
  QMutexLocker lock(&m_paramsMutex);
  const auto savedir = m_imageSaveDir;
  m_params.serializeToFile(savedir / "params.json");
  m_ioparams.serializeToFile(savedir / "ioparams.json");
}

namespace {

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
    beamform(rf, rfBeamformed, params.beamformerType);
    beamform_ms = timeit.get_ms();
  }

  // Recon
  {
    uspam::TimeIt timeit;

    // compute filter kernels
    const auto kernel = [&] {
      constexpr int numtaps = 95;
      if constexpr (std::is_same_v<T, double>) {
        return uspam::signal::firwin2(numtaps, params.filterFreq,
                                      params.filterGain);
      } else {
        const auto _kernel = uspam::signal::firwin2(numtaps, params.filterFreq,
                                                    params.filterGain);
        const auto kernel = arma::conv_to<arma::Col<T>>::from(_kernel);
        return kernel;
      }
    }();

    // Apply filter and envelope
    const cv::Range range(0, static_cast<int>(rf.n_cols));
    for (int i = range.start; i < range.end; ++i) {
      const auto _rf = rf.unsafe_col(i);
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

} // namespace

void DataProcWorker::processCurrentFrame() {
  PerformanceMetrics perfMetrics{};
  uspam::TimeIt timeit;

  // Init buffers in m_data
  initDataBuffers();

  // Read next RF scan from file
  {
    const uspam::TimeIt timeit;
    m_data->rf = m_loader.get<FloatType>(m_frameIdx);
    perfMetrics.fileloader_ms = timeit.get_ms();
  }

  const auto [paramsPA, paramsUS] = [&] {
    // Estimate background from current RF
    const arma::Col<FloatType> background_aline = arma::mean(m_data->rf, 1);

    // this->params and this->ioparams are used in this block
    // lock with paramsMutex
    QMutexLocker lock(&m_paramsMutex);
    {
      // Split RF into PA and US scan lines
      const uspam::TimeIt timeit;
      m_ioparams.splitRfPAUS_sub(m_data->rf, background_aline, m_data->PA.rf,
                                 m_data->US.rf);
      perfMetrics.splitRf_ms = timeit.get_ms();
    }

    return std::tuple(m_params.PA, m_params.US);
  }();

  const bool flip = uspam::recon::ReconParams::flip(m_frameIdx);

  constexpr bool USE_ASYNC = true;
  if constexpr (USE_ASYNC) {
    auto a1 = std::async(std::launch::async, procOne<FloatType>,
                         std::ref(paramsPA), std::ref(m_data->PA), flip);

    auto a2 = std::async(std::launch::async, procOne<FloatType>,
                         std::ref(paramsUS), std::ref(m_data->US), flip);

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
          procOne<FloatType>(paramsPA, m_data->PA, flip);
      perfMetrics.beamform_ms = beamform_ms;
      perfMetrics.recon_ms = recon_ms;
      perfMetrics.imageConversion_ms = imageConversion_ms;
    }

    {
      const auto [beamform_ms, recon_ms, imageConversion_ms] =
          procOne<FloatType>(paramsUS, m_data->US, flip);
      perfMetrics.beamform_ms += beamform_ms;
      perfMetrics.recon_ms += recon_ms;
      perfMetrics.imageConversion_ms += imageConversion_ms;
    }
  }

  // Compute scalebar scalar
  // fct is the depth [m] of one radial pixel
  m_data->fct = [&] {
    constexpr double soundSpeed = 1500.0; // [m/s] Sound speed
    constexpr double fs = 180e6;          // [1/s] Sample frequency

    // [m] multiplier to convert sampled US points to meters. 2x travel path
    constexpr double fctRect = soundSpeed / fs / 2;

    // [points]
    const auto USpoints_rect = static_cast<double>(m_data->US.rf.n_rows);

    // [points]
    const auto USpoints_radial =
        static_cast<double>(m_data->US.radial.rows) / 2;

    // [m]
    const auto fctRadial = fctRect * USpoints_rect / USpoints_radial;
    return fctRadial;
  }();

  {
    const uspam::TimeIt timeit;
    uspam::imutil::makeOverlay(m_data->US.radial, m_data->PA.radial,
                               m_data->PAUSradial);
    perfMetrics.makeOverlay_ms = timeit.get_ms();
  }

  m_data->PAUSradial_img = cvMatToQImage(m_data->PAUSradial);

  // Send images to GUI thread
  emit resultReady(m_data);
  emit frameIdxChanged(m_frameIdx);

  // Save to file
  {
    const uspam::TimeIt timeit;

    // USradial_img.save(
    //     path2QString(imageSaveDir / std::format("US_{:03d}.png", i)));
    // PAradial_img.save(
    //     path2QString(imageSaveDir / std::format("PA_{:03d}.png", i)));
    // PAUSradial_img.save(
    //     path2QString(imageSaveDir / std::format("PAUS_{:03d}.png", i)));

    auto *pool = QThreadPool::globalInstance();

    // using snprintf because apple clang doesn't support std::format yet...
    // NOLINTBEGIN(*-magic-numbers,*-pointer-decay,*-avoid-c-arrays)
    char _buf[64];
    std::snprintf(_buf, sizeof(_buf), "US_%03d.png", m_frameIdx);
    auto fname = path2QString(m_imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(m_data->US.radial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PA_%03d.png", m_frameIdx);
    fname = path2QString(m_imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(m_data->PA.radial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PAUS_%03d.png", m_frameIdx);
    fname = path2QString(m_imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(m_data->PAUSradial_img, fname));
    // NOLINTEND(*-magic-numbers,*-pointer-decay,*-avoid-c-arrays)

    perfMetrics.writeImages_ms = timeit.get_ms();
  }

  const auto elapsed = timeit.get_ms();

  auto msg = QString("Frame %1/%2 took %3 ms. ")
                 .arg(m_frameIdx)
                 .arg(m_loader.size())
                 .arg(static_cast<int>(elapsed));

  QTextStream stream(&msg);
  stream << perfMetrics;

  emit error(msg);
}
