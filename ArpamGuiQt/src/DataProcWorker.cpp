#include "DataProcWorker.hpp"
#include "strConvUtils.hpp"
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

namespace io = uspam::io;

namespace {

auto estimate_aline_background_from_file(const io::IOParams &ioparams,
                                         const fs::path &fname,
                                         int num_alines = 0) {
  const auto num_alines_all = ioparams.get_num_scans<uint16_t>(fname, 1);
  if (num_alines < 1) {
    num_alines = num_alines_all;
  } else {
    num_alines = std::min(num_alines, num_alines_all);
  }

  arma::Mat<uint16_t> rf(uspam::io::RF_ALINE_SIZE, num_alines,
                         arma::fill::none);
  ioparams.load_rf<uint16_t>(fname, rf, 1, 1, num_alines);

  arma::vec background = arma::mean(arma::conv_to<arma::mat>::from(rf), 1);
  return background;
}

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

struct ReconPerformanceStats {
  float reconTimeMs;
};

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

    // Init buffers
    {
      QMutexLocker lock(&m_paramsMutex);
      m_data.rfPair =
          m_ioparams.allocateSplitPair<FloatType>(m_loader.getAlinesPerBscan());
    }
    m_data.rfLog = io::PAUSpair<uint8_t>::zeros_like(m_data.rfPair);

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
void procOne(const uspam::recon::ReconParams &params, arma::Mat<T> &rf,
             arma::Mat<uint8_t> &rfLog, bool flip, cv::Mat &radial_img,
             QImage &radial_qimg) {

  uspam::recon::reconOneScan<T>(params, rf, rfLog, flip);
  radial_img = uspam::imutil::makeRadial(rfLog);
  radial_qimg = cvMatToQImage(radial_img);
}

class ImageWriteTask : public QRunnable {
  QImage img;
  QString fname;

public:
  ImageWriteTask(QImage img, QString fname)
      : img(std::move(img)), fname(std::move(fname)) {}
  void run() override { img.save(fname); }
};

struct PerformanceMetrics {
  float fileloader_ms{};
  float splitRfPAUS_ms{};
  float reconUSPA_ms{};
  float makeOverlay_ms{};
  float writeImages_ms{};

  [[nodiscard]] auto toString() const -> std::string {
    std::stringstream ss;
    ss << "fileloader " << static_cast<int>(fileloader_ms);
    ss << ", splitRfPAUS " << static_cast<int>(splitRfPAUS_ms);
    ss << ", reconUSPA " << static_cast<int>(reconUSPA_ms);
    ss << ", makeOverlay " << static_cast<int>(makeOverlay_ms);
    // ss << ", writeImages " << static_cast<int>(writeImages_ms);
    return ss.str();
  }
};

} // namespace

void DataProcWorker::processCurrentFrame() {
  const bool flip{m_frameIdx % 2 == 0};

  PerformanceMetrics perfMetrics{};
  uspam::TimeIt timeit;

  // Read next RF scan from file
  {
    const uspam::TimeIt timeit;
    m_loader.get(m_data.rf, m_frameIdx);
    perfMetrics.fileloader_ms = timeit.get_ms();
  }

  const auto [paramsPA, paramsUS] = [&] {
    // Estimate background from current RF
    const auto rf_f64 = arma::conv_to<arma::mat>::from(m_data.rf);
    const arma::vec background_aline = arma::mean(rf_f64, 1);

    // this->params and this->ioparams are used in this block
    // lock with paramsMutex
    QMutexLocker lock(&m_paramsMutex);
    {
      // Split RF into PA and US scan lines
      const uspam::TimeIt timeit;
      m_ioparams.splitRfPAUS_sub(rf_f64, background_aline, m_data.rfPair);
      perfMetrics.splitRfPAUS_ms = timeit.get_ms();
    }

    return std::tuple(m_params.PA, m_params.US);
  }();

  // Recon
  QImage USradial_img;
  QImage PAradial_img;
  cv::Mat USradial;
  cv::Mat PAradial;

  constexpr bool USE_ASYNC = true;
  if constexpr (USE_ASYNC) {
    const uspam::TimeIt timeit;

    const auto a1 =
        std::async(std::launch::async, procOne<FloatType>, std::ref(paramsPA),
                   std::ref(m_data.rfPair.PA), std::ref(m_data.rfLog.PA), flip,
                   std::ref(PAradial), std::ref(PAradial_img));

    const auto a2 =
        std::async(std::launch::async, procOne<FloatType>, std::ref(paramsUS),
                   std::ref(m_data.rfPair.US), std::ref(m_data.rfLog.US), flip,
                   std::ref(USradial), std::ref(USradial_img));

    a1.wait();
    a2.wait();

    perfMetrics.reconUSPA_ms = timeit.get_ms();
  } else {
    const uspam::TimeIt timeit;

    procOne<FloatType>(paramsPA, m_data.rfPair.PA, m_data.rfLog.PA, flip,
                       PAradial, PAradial_img);
    procOne<FloatType>(paramsUS, m_data.rfPair.US, m_data.rfLog.US, flip,
                       USradial, USradial_img);

    perfMetrics.reconUSPA_ms = timeit.get_ms();
  }

  // Compute scalebar scalar
  // fct is the depth [m] of one radial pixel
  const auto fct = [&] {
    constexpr double soundSpeed = 1500.0; // [m/s] Sound speed
    constexpr double fs = 180e6;          // [1/s] Sample frequency

    // [m] multiplier to convert sampled US points to meters. 2x travel path
    constexpr double fctRect = soundSpeed / fs / 2;

    // [points]
    const auto USpoints_rect = static_cast<double>(m_data.rfPair.US.n_rows);

    // [points]
    const auto USpoints_radial = static_cast<double>(USradial.rows) / 2;

    // [m]
    const auto fctRadial = fctRect * USpoints_rect / USpoints_radial;
    return fctRadial;
  }();

  cv::Mat PAUSradial; // CV_8U3C
  {
    const uspam::TimeIt timeit;
    uspam::imutil::makeOverlay(USradial, PAradial, PAUSradial);
    perfMetrics.makeOverlay_ms = timeit.get_ms();
  }

  QImage PAUSradial_img = cvMatToQImage(PAUSradial);

  // Send images to GUI thread
  emit resultReady(USradial_img, PAUSradial_img, fct);
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
    pool->start(new ImageWriteTask(USradial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PA_%03d.png", m_frameIdx);
    fname = path2QString(m_imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(PAradial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PAUS_%03d.png", m_frameIdx);
    fname = path2QString(m_imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(PAUSradial_img, fname));
    // NOLINTEND(*-magic-numbers,*-pointer-decay,*-avoid-c-arrays)

    perfMetrics.writeImages_ms = timeit.get_ms();
  }

  const auto elapsed = timeit.get_ms();

  auto msg = QString("Frame %1/%2 took %3 ms. ")
                 .arg(m_frameIdx)
                 .arg(m_loader.size())
                 .arg(static_cast<int>(elapsed));
  msg += perfMetrics.toString();

  emit error(msg);
}
