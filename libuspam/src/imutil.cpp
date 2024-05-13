#include "uspam/imutil.hpp"
#include <cassert>
#include <opencv2/opencv.hpp>

namespace uspam::imutil {

// US and PA are CV_8UC1, PAUS will be CV_8UC3
void makeOverlay(const cv::Mat &US, const cv::Mat &PA, cv::Mat &PAUS,
                 const uint8_t PAthresh) {
  assert(US.type() == CV_8UC1);
  assert(PA.type() == CV_8UC1);

  // Convert US to 3 channel, save to output mat
  cv::cvtColor(US, PAUS, cv::COLOR_GRAY2BGR);

  // Apply colormap to PA, save to intermediate
  cv::Mat PAclr;
  cv::applyColorMap(PA, PAclr, cv::COLORMAP_HOT);

  // Get positive PA mask
  cv::Mat mask;
  cv::threshold(PA, mask, PAthresh, 1, cv::THRESH_BINARY);

  // Copy positive PA to output array
  PAclr.copyTo(PAUS, mask);
}
} // namespace uspam::imutil