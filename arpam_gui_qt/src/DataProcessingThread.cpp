#include "DataProcessingThread.hpp"
#include <QtDebug>
#include <QtLogging>
#include <armadillo>
#include <chrono>
#include <filesystem>
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

void DataProcessingThread::setBinfile(const QString &binfile) {
  qInfo() << "DataProcessingThread set currentBinfile to" << binfile;
  currentBinfile = binfile;
  _ready = true;
  _abortCurrent = false;
  _condition.wakeOne();
}

void DataProcessingThread::run() {
  QMutexLocker locker(&_mutex);

  while (!_abortCurrent) {
    if (!_ready) {
      _condition.wait(&_mutex); // Wait until a task is set
    }
    if (_shouldStopThread) {
      break;
    }

    locker.unlock();

    this->processCurrentBinfile();
    emit finishedOneFile();
  }
}

void DataProcessingThread::stopCurentWork() {
  _abortCurrent = true;
  _condition.wakeOne();
}

void DataProcessingThread::threadShouldStop() {
  _abortCurrent = true;
  _shouldStopThread = true;
  _condition.wakeOne();
}

void DataProcessingThread::processCurrentBinfile() {
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

  try {
    uspam::io::BinfileLoader<uint16_t> loader(ioparams, binpath);

    const arma::vec background_aline =
        estimate_aline_background_from_file(ioparams, binpath, 1000);
    const auto background = ioparams.splitRfPAUS_aline(background_aline);

    arma::Mat<uint16_t> rf(uspam::io::RF_ALINE_SIZE, 1000, arma::fill::none);
    auto rfPair = ioparams.allocateSplitPair<double>(1000);
    auto rfLog = io::PAUSpair<double>::zeros_like(rfPair);

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

      // Background subtraction
      rfPair.US.each_col() -= background.US.col(0);

      // Recon
      params.reconOneScan(rfPair, rfLog, flip);

      // Results here are 64F
      const cv::Mat PAradial = uspam::imutil::makeRadial(rfLog.PA);
      const cv::Mat USradial = uspam::imutil::makeRadial(rfLog.US);

      cv::Mat PAradial_normalize = PAradial * 255;
      cv::Mat PAradial_u8;
      PAradial_normalize.convertTo(PAradial_u8, CV_8U);

      cv::Mat USradial_normalize = USradial * 255;
      cv::Mat USradial_u8;
      USradial_normalize.convertTo(USradial_u8, CV_8U);

      const QImage PAradial_img = cvMatToQImage(PAradial_u8);
      const QImage USradial_img = cvMatToQImage(USradial_u8);

      emit resultReady(PAradial_img, USradial_img);

      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::high_resolution_clock::now() - start_time)
              .count();

      emit error(QString("Processed image %1. Took %2 ms").arg(i).arg(elapsed));

      ++i;
    }

  } catch (const std::runtime_error &e) {
    auto msg = QString::fromStdString(e.what());
    qWarning() << "DataProcessingThread runtime_error: " << msg;
    emit error(msg);
  }
}