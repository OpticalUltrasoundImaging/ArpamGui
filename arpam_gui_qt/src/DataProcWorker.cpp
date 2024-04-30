#include "DataProcWorker.hpp"
#include <QtDebug>
#include <QtLogging>
#include <armadillo>
#include <chrono>
#include <filesystem>
#include <future>
#include <uspam/timeit.hpp>
#include <uspam/uspam.hpp>

namespace fs = std::filesystem;
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
    QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
    return image.copy(); // Use copy to detach from original data
  }
  // 8-bit, 3 channel
  case CV_8UC3: {
    QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
    return image.rgbSwapped(); // Convert BGR to RGB
  }
  // 8-bit, 1 channel
  case CV_8UC1: {
    QImage image(mat.data, mat.cols, mat.rows, mat.step,
                 QImage::Format_Grayscale8);
    return image.copy();
  }
  // 64F, 1 channel
  case CV_64FC1: {
    cv::Mat mat_normalized = mat * 255;
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
  currentBinfile = binfile;
  doPostProcess();
}

void DataProcWorker::doPostProcess() {
  _abortCurrent = false;
  _ready = false;

  try {
    this->processCurrentBinfile();
  } catch (const std::runtime_error &e) {
    const auto msg = QString::fromStdString(e.what());
    qWarning() << "processCurrentBinfile exception: " << msg;
    emit error("processCurrentBinfile exception: " + msg);
  }

  _abortCurrent = false;
  _ready = true;

  emit finishedOneFile();
}

void DataProcWorker::abortCurrentWork() {
  if (!_ready) {
    _abortCurrent = true;
  }
}

namespace {

void procOne(const uspam::recon::ReconParams &params,
             const arma::Mat<double> &background, arma::Mat<double> &rf,
             arma::Mat<uint8_t> &rfLog, bool flip, QImage &radial_qimg) {
  rf.each_col() -= background.col(0);
  params.reconOneScan(rf, rfLog, flip);
  const cv::Mat PAradial = uspam::imutil::makeRadial(rfLog);
  radial_qimg = cvMatToQImage(PAradial);
}

} // namespace

void DataProcWorker::processCurrentBinfile() {
  qDebug() << "Processing binfile: " << currentBinfile;
  const fs::path binpath = qString2Path(currentBinfile);
  const fs::path savedir = binpath.parent_path() / "images";

  {
    QString msg = tr("binpath ") + path2QString(binpath);
    qInfo() << msg;
  }

  if (!fs::create_directory(savedir) && !fs::exists(savedir)) {
    emit error(tr("Failed to create savedir") + path2QString(savedir));
  }

  const auto ioparams = uspam::io::IOParams::system2024v1();
  const auto params = uspam::recon::ReconParams2::system2024v1();
  const auto paramsPA = params.getPA();
  const auto paramsUS = params.getUS();

  uspam::io::BinfileLoader<uint16_t> loader(ioparams, binpath);

  const arma::vec background_aline =
      estimate_aline_background_from_file(ioparams, binpath, 1000);
  const auto background = ioparams.splitRfPAUS_aline(background_aline);

  arma::Mat<uint16_t> rf(uspam::io::RF_ALINE_SIZE, 1000, arma::fill::none);
  auto rfPair = ioparams.allocateSplitPair<double>(1000);
  auto rfLog = io::PAUSpair<uint8_t>::zeros_like(rfPair);

  const int starti = 0;
  const int nscans = loader.size();
  const int endi = starti + nscans;
  int i = 0;

  while (!_abortCurrent && i < endi) {
    const bool flip{i % 2 != 0};

    const auto start_time = std::chrono::high_resolution_clock::now();

    // Read next RF scan from file
    loader.getNext(rf);

    // Split RF into PA and US scan lines
    ioparams.splitRfPAUS(rf, rfPair);

    // Recon
    // rfPair.PA.each_col() -= background.PA.col(0);
    // paramsPA.reconOneScan(rfPair.PA, rfLog.PA, flip);
    // const cv::Mat PAradial = uspam::imutil::makeRadial(rfLog.PA);
    // const QImage PAradial_img = cvMatToQImage(PAradial);

    // rfPair.US.each_col() -= background.US.col(0);
    // paramsUS.reconOneScan(rfPair.US, rfLog.US, flip);
    // const cv::Mat USradial = uspam::imutil::makeRadial(rfLog.US);
    // const QImage USradial_img = cvMatToQImage(USradial);
    QImage USradial_img, PAradial_img;

    // procOne(paramsPA, background.PA, rfPair.PA, rfLog.PA, flip,
    // PAradial_img);

    // procOne(paramsUS, background.US, rfPair.US, rfLog.US, flip,
    // USradial_img);

    const auto a1 =
        std::async(std::launch::async, procOne, std::ref(paramsPA),
                   std::ref(background.PA), std::ref(rfPair.PA),
                   std::ref(rfLog.PA), flip, std::ref(PAradial_img));

    const auto a2 =
        std::async(std::launch::async, procOne, std::ref(paramsUS),
                   std::ref(background.US), std::ref(rfPair.US),
                   std::ref(rfLog.US), flip, std::ref(USradial_img));

    a1.wait();
    a2.wait();
    emit resultReady(PAradial_img, USradial_img);

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time)
            .count();

    emit error(QString("Processed image %1/%2. Took %3 ms")
                   .arg(i)
                   .arg(endi)
                   .arg(elapsed));

    ++i;
  }

  if (_abortCurrent) {
    emit error("Aborted.");
  } else {
    emit error("processCurrentBinfile finished.");
  }
}