#pragma once

#include <armadillo>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>

namespace uspam::imutil {

// NOLINTBEGIN(*-magic-numbers)
template <typename T>
auto makeRectangular(const arma::Mat<T> &mat, int width = 640,
                     int height = 1000) {
  // Create a cv::mat that uses the same data
  // NOLINTNEXTLINE(*-casting)
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, cv::traits::Type<T>::value,
                 (void *)mat.memptr());
  cv::resize(cv_mat, cv_mat, {width, height});
  cv_mat.convertTo(cv_mat, CV_8U, 255.0);
  return cv_mat;
}
// NOLINTEND(*-magic-numbers)

template <typename T>
auto makeRadial(const arma::Mat<T> &mat, int final_size = 0) {
  // NOLINTNEXTLINE(*-casting)
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, cv::traits::Type<T>::value,
                 (void *)mat.memptr());

  const int r = std::min(cv_mat.rows, cv_mat.cols);
  const cv::Size dsize{r, r};
  const cv::Point2f center{static_cast<float>(r) / 2,
                           static_cast<float>(r) / 2};
  const auto maxRadius{static_cast<double>(r) / 2};

  cv::warpPolar(cv_mat, cv_mat, dsize, center, maxRadius,
                cv::WARP_INVERSE_MAP | cv::WARP_FILL_OUTLIERS);
  cv::rotate(cv_mat, cv_mat, cv::ROTATE_90_COUNTERCLOCKWISE);

  if (final_size != 0) {
    cv::resize(cv_mat, cv_mat, {final_size, final_size});
  }
  return cv_mat;
}

template <typename T>
auto makeRadial_v2(const arma::Mat<T> &mat, int padding = 0) {
  // NOLINTNEXTLINE(*-casting)
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, cv::traits::Type<T>::value,
                 (void *)mat.memptr());

  if (padding != 0) {
    cv::copyMakeBorder(cv_mat, cv_mat, 0, 0, padding, 0, 0);
  }

  const int r = std::min(cv_mat.rows, cv_mat.cols);
  const cv::Size dsize{r, r};
  const cv::Point2f center{static_cast<float>(r) / 2,
                           static_cast<float>(r) / 2};
  const auto maxRadius{static_cast<double>(r) / 2};

  cv::warpPolar(cv_mat, cv_mat, dsize, center, maxRadius,
                cv::WARP_INVERSE_MAP | cv::WARP_FILL_OUTLIERS);
  cv::rotate(cv_mat, cv_mat, cv::ROTATE_90_COUNTERCLOCKWISE);

  return cv_mat;
}

// equivalent to arma::fliplr but inplace
template <typename T> void fliplr_inplace(arma::Mat<T> &mat) {
  const int n = mat.n_cols;
  for (int j = 0; j < n / 2; ++j) {
    mat.swap_cols(j, n - 1 - j);
  }
}

// Make PAUS overlay image.
// US and PA are CV_8UC1, PAUS will be CV_8UC3
void makeOverlay(const cv::Mat &US, const cv::Mat &PA, cv::Mat &PAUS,
                 uint8_t PAthresh = 10);

} // namespace uspam::imutil

template <typename T> void medianBlur(arma::Mat<T> &mat, int ksize) {
  // NOLINTBEGIN
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, cv::traits::Type<T>::value,
                 (void *)mat.memptr());
  // NOLINTEND

  ksize = ksize % 2 == 0 ? ksize + 1 : ksize; // Ensure odd
  cv::medianBlur(cv_mat, cv_mat, ksize);

  mat = arma::Mat<T>(cv_mat.ptr<T>(), mat.n_rows, mat.n_cols, true);
}