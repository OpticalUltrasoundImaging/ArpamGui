#pragma once

#include "uspam/fft.hpp"
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <span>

namespace uspam::surface {

/*
Detect the position of the first strong rise from the center line using the
CUSUM algorithm
*/
template <Floating T>
[[nodiscard]] auto cusum(const std::span<const T> data, const T threshold,
                         const T drift) -> int {
  T pos{};
  for (int i = 0; i < data.size(); ++i) {
    pos = std::max<T>(0, pos + data[i] - drift);
    if (pos > threshold) {
      return i;
    }
  }
  return {};
}

/*
Detect the surface in every A-scan stored in `rf` and write results to `surface`
*/
template <Floating T>
auto findSurface(const arma::Mat<T> &rf, std::span<float> surface,
                 const float threshold, const float drift) {
  cv::Range range(0, rf.n_cols);
  cv::parallel_for_(range, [&](const cv::Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      surface[i] = cusum<T>(rf.unsafe_col(i), threshold, drift);
    }
  });

  // TODO fix missing
  auto v = surface[0];
}

template <typename T>
inline auto warp_inverse(T rho, T phi, cv::Point2f center, cv::Size2f dsize,
                         T maxRadius) {
  const T Kangle = dsize.height / T(CV_2PI);
  const T Klin = dsize.width / maxRadius;
  const T angle = phi / Kangle;
  const T magnitude = rho / Klin;

  const auto x = std::round(center.x + magnitude * std::cos(angle));
  const auto y = std::round(center.y + magnitude * std::sin(angle));
  return cv::Point2f{x, y};
}

} // namespace uspam::surface
