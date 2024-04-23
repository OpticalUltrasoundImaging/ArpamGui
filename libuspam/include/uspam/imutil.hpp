#pragma once

#include "uspam/io.hpp"
#include <armadillo>
#include <opencv2/opencv.hpp>

namespace uspam::imutil {

inline auto makeRectangular(const arma::mat &mat) {
  // Create a cv::mat that uses the same data
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, CV_64F, (void *)mat.memptr());
  cv::resize(cv_mat, cv_mat, {640, 1000});
  cv_mat.convertTo(cv_mat, CV_8U, 255.0);
  return cv_mat;
}

inline auto makeRadial(const arma::mat &mat, int final_size = 0) {
  cv::Mat cv_mat(mat.n_cols, mat.n_rows, CV_64F, (void *)mat.memptr());
  const int r = std::min(cv_mat.rows, cv_mat.cols);
  cv::resize(cv_mat, cv_mat, {r * 2, r * 2});
  cv::warpPolar(cv_mat, cv_mat, {r * 2, r * 2}, {(float)r, (float)r}, (double)r,
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

} // namespace uspam::imutil
