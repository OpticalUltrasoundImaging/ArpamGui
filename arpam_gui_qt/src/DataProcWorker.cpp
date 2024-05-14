#include "DataProcWorker.hpp"
#include <QThreadPool>
#include <QtDebug>
#include <QtLogging>
#include <armadillo>
#include <chrono>
#include <cstdio>
#include <format>
#include <future>
#include <sstream>
#include <tuple>
#include <uspam/imutil.hpp>
#include <uspam/timeit.hpp>
#include <uspam/uspam.hpp>
#include <utility>

namespace io = uspam::io;

namespace {

// Convert a QString to a fs::path
auto qString2Path(const QString &str) {
  const auto utf8array = str.toUtf8();
  return fs::path(utf8array.constData());
}

auto path2string(const fs::path &path) {
#if defined(_WIN32) || defined(_WIN64)
  return path.wstring();
#else
  return path.string();
#endif
}

auto path2QString(const fs::path &path) {
#if defined(_WIN32) || defined(_WIN64)
  return QString::fromStdWString(path.wstring());
#else
  return QString::fromStdString(path.string());
#endif
}

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

void DataProcWorker::setBinfile(const QString &binfile) {
  qInfo() << "DataProcWorker set currentBinfile to" << binfile;
  binfilePath = qString2Path(binfile);
  imageSaveDir = binfilePath.parent_path() / binfilePath.stem();

  qInfo() << "Processing binfile: " << path2QString(binfilePath);

  if (!fs::create_directory(imageSaveDir) && !fs::exists(imageSaveDir)) {
    emit error(tr("Failed to create imageSaveDir ") +
               path2QString(imageSaveDir));
  } else {
    emit error(tr("Saving images to ") + path2QString(imageSaveDir));
  }

  try {
    // Init loader
    loader.setParams(ioparams);
    loader.open(binfilePath);
    emit updateMaxFrames(loader.size());

    // Init buffers
    {
      QMutexLocker lock(&paramsMutex);
      rfPair = ioparams.allocateSplitPair<double>(loader.getAlinesPerBscan());
    }
    rfLog = io::PAUSpair<uint8_t>::zeros_like(rfPair);

    // Start processing
    this->play();
  } catch (const std::runtime_error &e) {
    const auto msg = QString::fromStdString(e.what());
    qWarning() << "DataProcWorker exception: " << msg;
    emit error("DataProcWorker exception: " + msg);
  }

  emit finishedOneFile();
}

void DataProcWorker::play() {
  _isPlaying = true;

  while (_isPlaying && frameIdx < loader.size()) {
    playOne(frameIdx);
    frameIdx++;
  }

  if (_isPlaying) {
    emit error("DataProcWorker::play Finished.");
  } else {
    emit error("DataProcWorker::play Paused.");
  }
  _isPlaying = false;

  emit finishedOneFile();
}

void DataProcWorker::playOne(int idx) {
  frameIdx = idx;

  processCurrentFrame();
}
void DataProcWorker::replayOne() { processCurrentFrame(); }

void DataProcWorker::pause() { _isPlaying = false; }

namespace {

void procOne(const uspam::recon::ReconParams &params, arma::Mat<double> &rf,
             arma::Mat<uint8_t> &rfLog, bool flip, cv::Mat &radial_img,
             QImage &radial_qimg) {
  params.reconOneScan(rf, rfLog, flip);
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
  const bool flip{frameIdx % 2 == 0};

  PerformanceMetrics perfMetrics{};
  uspam::TimeIt timeit;

  // Read next RF scan from file
  {
    const uspam::TimeIt timeit;
    loader.get(rf, frameIdx);
    perfMetrics.fileloader_ms = timeit.get_ms();
  }

  const auto [paramsPA, paramsUS] = [&] {
    // Estimate background from current RF
    const auto rf_f64 = arma::conv_to<arma::mat>::from(rf);
    const arma::vec background_aline = arma::mean(rf_f64, 1);

    // this->params and this->ioparams are used in this block
    // lock with paramsMutex
    QMutexLocker lock(&paramsMutex);
    {
      // Split RF into PA and US scan lines
      const uspam::TimeIt timeit;
      ioparams.splitRfPAUS_sub(rf_f64, background_aline, rfPair);
      perfMetrics.splitRfPAUS_ms = timeit.get_ms();
    }

    const auto paramsPA = params.getPA();
    const auto paramsUS = params.getUS();
    return std::tuple(paramsPA, paramsUS);
  }();

  // Recon
  // rfPair.PA.each_col() -= background.PA.col(0);
  // paramsPA.reconOneScan(rfPair.PA, rfLog.PA, flip);
  // const cv::Mat PAradial = uspam::imutil::makeRadial(rfLog.PA);
  // const QImage PAradial_img = cvMatToQImage(PAradial);

  // rfPair.US.each_col() -= background.US.col(0);
  // paramsUS.reconOneScan(rfPair.US, rfLog.US, flip);
  // const cv::Mat USradial = uspam::imutil::makeRadial(rfLog.US);
  // const QImage USradial_img = cvMatToQImage(USradial);
  QImage USradial_img;
  QImage PAradial_img;
  cv::Mat USradial;
  cv::Mat PAradial;

  // procOne(paramsPA, rfPair.PA, rfLog.PA, flip,
  // PAradial_img);

  // procOne(paramsUS, rfPair.US, rfLog.US, flip,
  // USradial_img);

  {
    const uspam::TimeIt timeit;

    const auto a1 = std::async(std::launch::async, procOne, std::ref(paramsPA),
                               std::ref(rfPair.PA), std::ref(rfLog.PA), flip,
                               std::ref(PAradial), std::ref(PAradial_img));

    const auto a2 = std::async(std::launch::async, procOne, std::ref(paramsUS),
                               std::ref(rfPair.US), std::ref(rfLog.US), flip,
                               std::ref(USradial), std::ref(USradial_img));

    a1.wait();
    a2.wait();

    perfMetrics.reconUSPA_ms = timeit.get_ms();
  }

  cv::Mat PAUSradial; // CV_8U3C
  {
    const uspam::TimeIt timeit;
    uspam::imutil::makeOverlay(USradial, PAradial, PAUSradial);
    perfMetrics.makeOverlay_ms = timeit.get_ms();
  }

  QImage PAUSradial_img = cvMatToQImage(PAUSradial);

  // Send images to GUI thread
  emit resultReady(USradial_img, PAUSradial_img);
  emit updateFrameIdx(frameIdx);

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
    std::snprintf(_buf, sizeof(_buf), "US_%03d.png", frameIdx);
    auto fname = path2QString(imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(USradial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PA_%03d.png", frameIdx);
    fname = path2QString(imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(PAradial_img, fname));

    std::snprintf(_buf, sizeof(_buf), "PAUS_%03d.png", frameIdx);
    fname = path2QString(imageSaveDir / std::string(_buf));
    pool->start(new ImageWriteTask(PAUSradial_img, fname));
    // NOLINTEND(*-magic-numbers,*-pointer-decay,*-avoid-c-arrays)

    perfMetrics.writeImages_ms = timeit.get_ms();
  }

  const auto elapsed = timeit.get_ms();

  auto msg = QString("Frame %1/%2 took %3 ms. ")
                 .arg(frameIdx)
                 .arg(loader.size())
                 .arg(static_cast<int>(elapsed));
  msg += perfMetrics.toString();

  emit error(msg);
}