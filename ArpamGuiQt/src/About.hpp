#pragma once

#include <QString>
#include <armadillo>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <rapidjson/rapidjson.h>
#include <sstream>
#include <string>

struct ARPAM_GUI_ABOUT {
  QString operator()() {
    std::stringstream ss;

    ss << "Qt: " << QT_VERSION_STR << "\n";

    ss << "Armadillo: " << ARMA_VERSION_MAJOR << "." << ARMA_VERSION_MINOR
       << "." << ARMA_VERSION_PATCH << "\n";

    ss << "OpenCV: " << cv::getVersionString() << "\n";

    // ss << "FFTW3: " << fftw_version << "\n";

    // ss << "RapidJSON: " << RAPIDJSON_VERSION_STRING << "\n";

    return QString::fromStdString(ss.str());
  }
};