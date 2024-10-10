#include "About.hpp"
#include "DAQ/DAQ.hpp"
#include "Motor/NI.hpp"
#include <QMessageBox>
#include <armadillo>
#include <fftw3.h>
#include <kfr/kfr.h>
#include <oneapi/tbb/version.h>
#include <opencv2/opencv.hpp>
#include <qcustomplot.h>
#include <rapidjson/rapidjson.h>
#include <sstream>

namespace arpam_about {

auto aboutString() -> QString {
  std::stringstream ss;

  ss << "ArpamGuiQt (Build with " << ARPAM_COMPILER_NAME << " "
     << ARPAM_COMPILER_VERSION << " on " << ARPAM_COMPILE_DATE "-"
     << ARPAM_COMPILE_TIME << ")\n\n";

  ss << "-- Bundled Dependencies\n";
  ss << "Qt: " << QT_VERSION_STR << " | ";
  ss << "QCustomPlot: " << QCUSTOMPLOT_VERSION_STR << " | ";

  ss << "Armadillo: " << ARMA_VERSION_MAJOR << "." << ARMA_VERSION_MINOR << "."
     << ARMA_VERSION_PATCH << " | ";

  ss << "OpenCV: " << cv::getVersionString() << " | ";

  ss << "KFR: " << KFR_VERSION_STRING << " | ";

#if defined(_WIN32) || defined(_WIN64)
  // Temp fix for Windows
  // https://github.com/microsoft/vcpkg/issues/39719
  ss << "FFTW3: unknown"
     << " | ";
#else
  // NOLINTNEXTLINE(*-pointer-decay)
  ss << "FFTW3: " << fftw_version << " | ";
#endif

  ss << "RapidJSON: " << RAPIDJSON_VERSION_STRING << " | ";

  ss << "oneTBB: " << TBB_VERSION_STRING << "\n";
  ss << "-- Bundled Dependencies\n\n";

  ss << "-- Data acquisition interface";
#ifdef ARPAM_HAS_ALAZAR
  ss << "\n";
  ss << daq::getDAQInfo();
  ss << "-- Data acquisition interface\n\n";
#else
  ss << ": No DAQ support.\n";
#endif

  ss << "-- Motor control interface";
#ifdef ARPAM_HAS_NI
  ss << "\n";
  ss << motor::getNIDAQInfo();
  ss << "-- Motor control interface\n";
#else
  ss << ": No motor control support.\n";
#endif

  return QString::fromStdString(ss.str());
}

void showAboutDialog(QWidget *parent) {
  QMessageBox::about(parent, "About ArpamGui", aboutString());
}

} // namespace arpam_about
