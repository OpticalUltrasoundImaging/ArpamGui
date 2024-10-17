#include "RFBuffer.hpp"
#include "Common.hpp"
#include "strConvUtils.hpp"
#include <QList>
#include <fstream>
#include <future>
#include <uspam/imutil.hpp>
#include <uspam/timeit.hpp>

template <uspam::Floating T>
void BScanData_<T>::saveBScanData(const fs::path &directory,
                                  const std::string &prefix,
                                  const ExportSetting &exportSetting) const {
  // uspam::TimeIt<true> timeit("saveBScanData");

  if (exportSetting.saveRF) {
    // Save env
    const auto envPath = (directory / (prefix + "env.bin")).string();
    rfEnv.save(envPath, arma::raw_binary);

    // Save rf
    const auto rfPath = (directory / (prefix + "rf.bin")).string();
    rf.save(rfPath, arma::raw_binary);
  }

  if (exportSetting.saveRadialImages) {
    // Save radial
    const auto radialPath = (directory / (prefix + "radial.tiff")).string();
    radial_img.save(path2QString(radialPath));
  }

  if (exportSetting.saveRectImages) {
    const auto rectPath = (directory / (prefix + "rect.tiff")).string();
    const auto rectImage = uspam::imutil::armaMatToCvMat(rfLog);
    cv::Mat img;
    cv::resize(rectImage, img, {(int)rfLog.n_cols, (int)rfLog.n_cols});
    cv::imwrite(rectPath, img);
  }
}

template void
BScanData_<ArpamFloat>::saveBScanData(const fs::path &directory,
                                      const std::string &prefix,
                                      const ExportSetting &exportSetting) const;

QImage cropImage(const QImage &image, const QRect &rect) {
  // Ensure the QRect is valid within the bounds of the image
  QRect validRect = rect.intersected(image.rect());

  // Crop the image
  return image.copy(validRect);
}

// Return a crop of the image. (no copy)
cv::Mat cropImage(const cv::Mat &image, const cv::Rect &rect) {
  // Ensure the QRect is valid within the bounds of the image
  cv::Rect validRect =
      rect & cv::Rect(0, 0, image.cols,
                      image.rows); // Intersection of ROI and image bounds

  // Crop the image using the valid region of interest
  return image(validRect);
}

auto cvRect(const QRect &qrect) {
  return cv::Rect(qrect.x(), qrect.y(), qrect.width(), qrect.height());
}

void exportImageList(
    const std::vector<std::pair<QImage, std::string>> &imageList,
    const fs::path &savedir) {

  fs::create_directories(savedir);

  for (const auto &[image, name] : imageList) {
    const auto path = savedir / name;

    if (!image.save(path2QString(path))) {
      std::cerr << "Failed to save image to path: " << path << "\n";
    }
  }
}

void exportImageList(
    const std::vector<std::pair<cv::Mat, std::string>> &imageList,
    const fs::path &savedir) {

  fs::create_directories(savedir);

  for (const auto &[image, name] : imageList) {
    const auto path = savedir / name;

    if (!cv::imwrite(path.c_str(), image)) {
      std::cerr << "Failed to save image to path: " << path << "\n";
    }
  }
}

// Used when endCol > n_cols. Handles wrap around
arma::Mat<uint8_t> boundaryCrop(const arma::Mat<uint8_t> &mat, int startCol,
                                int endCol) {
  // return mat.cols(startCol, mat.n_cols - 1);
  // return mat.head_cols(endCol - mat.n_cols);
  return arma::join_horiz(mat.cols(startCol, mat.n_cols - 1),
                          mat.head_cols(endCol - mat.n_cols));
}

template <uspam::Floating T>
void BScanData<T>::exportToFile(
    const fs::path &directory, const QList<annotation::Annotation> &annotations,
    const ExportSetting &exportSetting) const {
  if (!fs::exists(directory)) {
    fs::create_directory(directory);
  }

  // Save frame index
  // Touch file to create an empty txt file with the frame idx as title
  { std::ofstream fs(directory / fmt::format("frame_{}.txt", frameIdx)); }

  /*
  Exported crops from annotation
  Names should have the format
  "{modality}-{type_and_coord}-{label}.tiff"
  */
  auto aCrops = std::async(
      std::launch::async, &BScanData<T>::exportAnnotatedCrops, this,
      directory / "roi", std::ref(annotations), std::ref(exportSetting));

  // Save PA and US buffers/images
  auto aPA = std::async(std::launch::async, &BScanData_<T>::saveBScanData, &PA,
                        std::ref(directory), "PA", std::ref(exportSetting));
  auto aUS = std::async(std::launch::async, &BScanData_<T>::saveBScanData, &US,
                        std::ref(directory), "US", std::ref(exportSetting));
  // PA.saveBScanData(directory, "PA", exportSetting);
  // US.saveBScanData(directory, "US", exportSetting);

  // Save RF to bin
  if (exportSetting.saveRF) {
    // Save raw RF
    const auto rfPath = (directory / "rf.bin").string();
    rf.save(rfPath, arma::raw_binary);
  }

  // Save radial images
  if (exportSetting.saveRadialImages) {
    // Save combined image
    auto pausPath = (directory / "PAUSradial.tiff").string();
    cv::imwrite(pausPath, PAUSradial);
  }

  aCrops.get();
  aUS.get();
  aPA.get();
}

template void BScanData<ArpamFloat>::exportToFile(
    const fs::path &directory, const QList<annotation::Annotation> &annotations,
    const ExportSetting &exportSetting) const;

template <uspam::Floating T>
void BScanData<T>::exportAnnotatedCrops(
    const fs::path &directory, const QList<annotation::Annotation> &annotations,
    const ExportSetting &exportSetting) const {
  // uspam::TimeIt<true> timeit("exportAnnotatedCrops");

  if (annotations.empty()) {
    return;
  }

  // Load annotations
  std::vector<std::pair<QImage, std::string>> croppedQImages;
  std::vector<std::pair<cv::Mat, std::string>> croppedImages;

  for (const auto &anno : annotations) {
    switch (anno.type) {
    case annotation::Annotation::Type::Rect: {
      const auto qrect = anno.rect().toRect();
      const auto rect = cvRect(qrect);

      const auto annoSuffix =
          fmt::format("rect_{},{}_{},{}-{}.png", qrect.top(), qrect.left(),
                      qrect.bottom(), qrect.right(), anno.name.toStdString());

      // Crop PAUSradial_img
      {
        const auto cropped = cropImage(this->PAUSradial_img, qrect);
        const auto name = fmt::format("PAUSradial-{}", annoSuffix);
        croppedQImages.emplace_back(cropped, name);
      }

      // Crop PA radial
      {
        const auto cropped = cropImage(this->PA.radial_img, qrect);
        const auto name = fmt::format("PAradial-{}", annoSuffix);
        croppedQImages.emplace_back(cropped, name);
      }

      // Crop US radial
      {
        const auto cropped = cropImage(this->US.radial_img, qrect);
        const auto name = fmt::format("USradial-{}", annoSuffix);
        croppedQImages.emplace_back(cropped, name);
      }
    }

    break;
    case annotation::Annotation::Fan: {
      const auto arc = anno.arc();

      // startAngle starts at 90 degrees
      const auto startAngle = -arc.startAngle + 90;
      const auto spanAngle = -arc.spanAngle;
      const auto n_cols = this->US.rfLog.n_cols;
      const auto angle2rect = [n_cols](const double angle) {
        return static_cast<int>(std::round(angle / 360.0 * n_cols));
      };
      auto startCol = angle2rect(startAngle);
      auto spanCol = angle2rect(spanAngle);

      if (startCol < 0) {
        startCol = startCol + n_cols;
      }

      auto endCol = startCol + spanCol;

      if (startCol >= endCol) {
        std::swap(startCol, endCol);
      }

      const auto annoSuffix =
          fmt::format("fan_{:.2f},{:.2f}_{},{}-{}.png", arc.startAngle,
                      arc.spanAngle, startCol, endCol, anno.name.toStdString());

      const auto exportCropped = [&](const arma::Mat<uint8_t> &mat,
                                     std::string name) {
        arma::Mat<uint8_t> cropped;
        if (endCol < n_cols) {
          cropped = mat.cols(startCol, endCol);
        } else {
          cropped = boundaryCrop(mat, startCol, endCol);
        }
        const auto croppedImg_ = uspam::imutil::armaMatToCvMat(cropped);

        cv::Mat croppedImg;
        cv::resize(croppedImg_, croppedImg, {(int)n_cols, croppedImg_.rows});

        croppedImages.emplace_back(croppedImg, std::move(name));
      };

      // Export cropped rect
      exportCropped(US.rfLog, fmt::format("US-{}", annoSuffix));
      exportCropped(PA.rfLog, fmt::format("PA-{}", annoSuffix));

    } break;
    case annotation::Annotation::Polygon:
    case annotation::Annotation::Line:
    case annotation::Annotation::Size:
      break;
    }
  }

  exportImageList(croppedQImages, directory);
  exportImageList(croppedImages, directory);
};

template void BScanData<ArpamFloat>::exportAnnotatedCrops(
    const fs::path &directory, const QList<annotation::Annotation> &annotations,
    const ExportSetting &exportSetting) const;