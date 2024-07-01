#pragma once

#include <algorithm>
#include <span>

template <typename T> struct FWHM {
  int peakIdx;
  int lowerIdx;
  int upperIdx;

  T maxY;
  T lowerX;
  T upperX;

  auto width() const { return upperX - lowerX; }
};

template <typename T>
auto calcFWHM(const std::span<const T> &x, const std::span<const T> &y) {
  const auto it = std::max_element(y.begin(), y.end());
  const int peakIdx = std::distance(y.begin(), it);
  const T maxY = *it;
  const T halfMax = maxY / 2.0;

  int lowerIdx = peakIdx;
  int upperIdx = peakIdx;

  // Search for half max crossing points
  while (lowerIdx > 0 && y[lowerIdx] > halfMax) {
    lowerIdx--;
  }

  while (upperIdx < y.size() - 1 && y[upperIdx] > halfMax) {
    upperIdx++;
  }

  return FWHM<T>{peakIdx, lowerIdx, upperIdx, maxY, x[lowerIdx], x[upperIdx]};
}
