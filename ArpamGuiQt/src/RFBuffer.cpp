#include "RFBuffer.hpp"
#include "Common.hpp"
#include "strConvUtils.hpp"
#include <QList>
#include <fstream>
#include <future>

template <uspam::Floating T>
void BScanData_<T>::saveBScanData(const fs::path &directory,
                                  const std::string &prefix,
                                  const ExportSetting &exportSetting) {

  if (exportSetting.radialImages) {
    // Save radial
    const auto radialPath = (directory / (prefix + "radial.tiff")).string();
    cv::imwrite(radialPath, radial);
  }

  if (exportSetting.rf) {
    // Save env
    const auto envPath = (directory / (prefix + "env.bin")).string();
    rfEnv.save(envPath, arma::raw_binary);

    // Save rf
    const auto rfPath = (directory / (prefix + "rf.bin")).string();
    rf.save(rfPath, arma::raw_binary);
  }
}

template void
BScanData_<ArpamFloat>::saveBScanData(const fs::path &directory,
                                      const std::string &prefix,
                                      const ExportSetting &exportSetting);

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

template <uspam::Floating T>
void BScanData<T>::exportToFile(
    const fs::path &directory, const QList<annotation::Annotation> &annotations,
    const ExportSetting &exportSetting) {
  if (!fs::exists(directory)) {
    fs::create_directory(directory);
  }

  /*
  Exported crops from annotation
  Names should have the format
  "{modality}-{type_and_coord}-{label}.tiff"
  */
  if (!annotations.empty()) {
    // Load annotations

    std::vector<std::pair<QImage, std::string>> croppedQImages;
    // std::vector<std::pair<cv::Mat, std::string>> croppedImages;

    for (const auto &anno : annotations) {
      switch (anno.type) {
      case annotation::Annotation::Type::Rect: {
        const auto rect = anno.rect().toRect();

        const auto annoSuffix =
            fmt::format("rect_{},{}_{},{}-{}.png", rect.top(), rect.left(),
                        rect.bottom(), rect.right(), anno.name.toStdString());

        // Crop PAUSradial_img
        {
          const auto cropped = cropImage(this->PAUSradial_img, rect);
          const auto name = fmt::format("PAUSradial-{}", annoSuffix);
          croppedQImages.emplace_back(cropped, name);
        }

        // Crop PA
        {
          const auto cropped = cropImage(this->PA.radial_img, rect);
          const auto name = fmt::format("PAradial-{}", annoSuffix);
          croppedQImages.emplace_back(cropped, name);
        }

        // Crop US
        {
          const auto cropped = cropImage(this->US.radial_img, rect);
          const auto name = fmt::format("USradial-{}", annoSuffix);
          croppedQImages.emplace_back(cropped, name);
        }
      }

      break;
      case annotation::Annotation::Fan:
      case annotation::Annotation::Polygon:
      case annotation::Annotation::Line:
      case annotation::Annotation::Size:
        break;
      }
    }

    exportImageList(croppedQImages, directory / "roi");
  }

  // Save PA and US buffers/images
  auto aPA = std::async(std::launch::async, &BScanData_<T>::saveBScanData, &PA,
                        std::ref(directory), "PA", exportSetting);
  auto aUS = std::async(std::launch::async, &BScanData_<T>::saveBScanData, &US,
                        std::ref(directory), "US", exportSetting);
  // PA.saveBScanData(directory, "PA", exportSetting);
  // US.saveBScanData(directory, "US", exportSetting);

  if (exportSetting.rf) {
    // Save raw RF
    const auto rfPath = (directory / "rf.bin").string();
    rf.save(rfPath, arma::raw_binary);
  }

  // Save frame index
  // Touch file to create an empty txt file with the frame idx as title
  { std::ofstream fs(directory / fmt::format("frame_{}.txt", frameIdx)); }

  if (exportSetting.radialImages) {
    // Save combined image
    auto pausPath = (directory / "PAUSradial.tiff").string();
    cv::imwrite(pausPath, PAUSradial);
  }

  aUS.get();
  aPA.get();
}

template void BScanData<ArpamFloat>::exportToFile(
    const fs::path &directory, const QList<annotation::Annotation> &annotations,
    const ExportSetting &exportSetting);