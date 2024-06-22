#pragma once

#include <armadillo>
#include <opencv2/opencv.hpp>

namespace uspam::imutil {

template <typename T> int getCvType();
template <> inline consteval int getCvType<double>() { return CV_64F; }
template <> inline consteval int getCvType<float>() { return CV_32F; }
template <> inline consteval int getCvType<uint8_t>() { return CV_8U; }

// NOLINTBEGIN(*-magic-numbers)
template <typename T>
auto makeRectangular(const arma::Mat<T> &mat, int width = 640,
                     int height = 1000) {
  // Create a cv::mat that uses the same data
  // NOLINTNEXTLINT
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, getCvType<T>(), (void *)mat.memptr());
  cv::resize(cv_mat, cv_mat, {width, height});
  cv_mat.convertTo(cv_mat, CV_8U, 255.0);
  return cv_mat;
}
// NOLINTEND(*-magic-numbers)

template <typename T>
auto makeRadial(const arma::Mat<T> &mat, int final_size = 0) {
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, getCvType<T>(), (void *)mat.memptr());

  const int r = std::min(cv_mat.rows, cv_mat.cols);
  const cv::Size dsize{r * 2, r * 2};
  const cv::Point2f center{static_cast<float>(r), static_cast<float>(r)};
  const auto maxRadius{static_cast<double>(r)};

  cv::resize(cv_mat, cv_mat, dsize);
  cv::warpPolar(cv_mat, cv_mat, dsize, center, maxRadius,
                cv::WARP_INVERSE_MAP | cv::WARP_FILL_OUTLIERS);
  cv::rotate(cv_mat, cv_mat, cv::ROTATE_90_COUNTERCLOCKWISE);

  final_size = final_size == 0 ? r : final_size;
  cv::resize(cv_mat, cv_mat, {final_size, final_size});
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
