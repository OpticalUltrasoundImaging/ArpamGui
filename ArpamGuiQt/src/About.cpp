#include "About.hpp"
#include <QMessageBox>
#include <armadillo>
#include <fftw3.h>
#include <oneapi/tbb/version.h>
#include <opencv2/opencv.hpp>
#include <qcustomplot.h>
#include <rapidjson/rapidjson.h>
#include <sstream>

namespace arpam_about {

auto aboutString() -> QString {
  std::stringstream ss;

  ss << "ArpamGuiQt\n\n";

  ss << "Build with " << ARPAM_COMPILER_NAME << " version "
     << ARPAM_COMPILER_VERSION << " on " << ARPAM_COMPILE_DATE " at "
     << ARPAM_COMPILE_TIME << "\n";

  ss << "Qt: " << QT_VERSION_STR << "\n";
  ss << "QCustomPlot: " << QCUSTOMPLOT_VERSION_STR << "\n";

  ss << "Armadillo: " << ARMA_VERSION_MAJOR << "." << ARMA_VERSION_MINOR << "."
     << ARMA_VERSION_PATCH << "\n";

  ss << "OpenCV: " << cv::getVersionString() << "\n";

#if defined(_WIN32) || defined(_WIN64)
// Temp fix for Windows 
// https://github.com/microsoft/vcpkg/issues/39719
#else
  // NOLINTNEXTLINE(*-pointer-decay)
  ss << "FFTW3: " << fftw_version << "\n";
#endif

  ss << "RapidJSON: " << RAPIDJSON_VERSION_STRING << "\n";

  ss << "oneTBB: " << TBB_VERSION_STRING << "\n";

  return QString::fromStdString(ss.str());
}

void showAboutDialog(QWidget *parent) {
  QMessageBox::about(parent, "About ArpamGui", aboutString());
}

} // namespace arpam_about
