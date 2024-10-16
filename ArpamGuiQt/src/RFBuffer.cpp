#include "RFBuffer.hpp"
#include "Common.hpp"
#include "strConvUtils.hpp"
#include <QList>
#include <fstream>
#include <future>

template <uspam::Floating T>
void BScanData_<T>::saveBScanData(const fs::path &directory,
                                  const std::string &prefix) {
  // Save radial
  const auto radialPath = (directory / (prefix + "radial.bmp")).string();
  cv::imwrite(radialPath, radial);

  // Save env
  const auto envPath = (directory / (prefix + "env.bin")).string();
  rfEnv.save(envPath, arma::raw_binary);

  // Save rf
  const auto rfPath = (directory / (prefix + "rf.bin")).string();
  rf.save(rfPath, arma::raw_binary);
}

template void BScanData_<ArpamFloat>::saveBScanData(const fs::path &directory,
                                                    const std::string &prefix);

QImage cropImage(const QImage &image, const QRect &rect) {
  // Ensure the QRect is valid within the bounds of the image
  QRect validRect = rect.intersected(image.rect());

  // Crop the image
  return image.copy(validRect);
}

void exportImageList(
    const std::vector<std::pair<QImage, std::string>> &imageList,
    const fs::path &savedir) {

  fs::create_directories(savedir);

  for (const auto &[image, name] : imageList) {
    const auto path = savedir / name;

    image.save(path2QString(path));
  }
}

template <uspam::Floating T>
void BScanData<T>::exportToFile(
    const fs::path &directory,
    const QList<annotation::Annotation> &annotations) {
  if (!fs::exists(directory)) {
    fs::create_directory(directory);
  }

  /*
  Exported crops from annotation
  Names should have the format
  "{modality}-{type_and_coord}-{label}.bmp"
  */
  if (!annotations.empty()) {
    // Load annotations
    std::vector<std::pair<QImage, std::string>> croppedImages;
    for (const auto &anno : annotations) {
      switch (anno.type) {
      case annotation::Annotation::Type::Rect: {
        const auto rect = anno.rect().toRect();
        const auto cropped = cropImage(this->PAUSradial_img, rect);

        const auto name = fmt::format("PAUSradial_img-rect_{},{}_{},{}-{}.bmp",
                                      rect.top(), rect.left(), rect.bottom(),
                                      rect.right(), anno.name.toStdString());

        croppedImages.emplace_back(cropped, name);
      }

      break;
      case annotation::Annotation::Fan:
      case annotation::Annotation::Polygon:
      case annotation::Annotation::Line:
      case annotation::Annotation::Size:
        break;
      }
    }

    exportImageList(croppedImages, directory / "roi");
  }

  // Save PA and US buffers/images
  auto aPA = std::async(std::launch::async, &BScanData_<T>::saveBScanData, &PA,
                        std::ref(directory), "PA");
  auto aUS = std::async(std::launch::async, &BScanData_<T>::saveBScanData, &US,
                        std::ref(directory), "US");
  // PA.saveBScanData(directory, "PA");
  // US.saveBScanData(directory, "US");

  // Save raw RF
  const auto rfPath = (directory / "rf.bin").string();
  rf.save(rfPath, arma::raw_binary);

  // Save frame index
  // Touch file to create an empty txt file with the frame idx as title
  { std::ofstream fs(directory / fmt::format("frame_{}.txt", frameIdx)); }

  // Save combined image

  auto pausPath = (directory / "PAUSradial.bmp").string();
  cv::imwrite(pausPath, PAUSradial);

  aUS.get();
  aPA.get();
}

template void BScanData<ArpamFloat>::exportToFile(
    const fs::path &directory,
    const QList<annotation::Annotation> &annotations);