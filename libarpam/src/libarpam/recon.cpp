#include "fftconv.hpp"

#include "libarpam/recon.hpp"
#include "libarpam/signal.hpp"

namespace arpam::recon {

void recon(const Eigen::MatrixXd &rf, Eigen::MatrixXd &output) {

  if (rf.cols() != output.cols() || rf.rows() != output.rows()) {
    throw std::invalid_argument("Sizes of rf and output are different.");
  }

  // Get FIR filter kernel
  const int numtaps = 65;
  Eigen::ArrayXd freq(4);
  freq << 0., 0.1, 0.3, 1.;
  Eigen::ArrayXd gain(4);
  gain << 0., 1., 1., 0.;

  const auto aline_size = rf.rows();
  //   const auto num_alines = rf.cols();

  pipeline::FIRFilter fir_filter(aline_size, numtaps, freq, gain);
  pipeline::EnvelopeDetection envelope_detection(aline_size);

  Eigen::ArrayXd buf1(aline_size);
  std::span<double> buf1span(buf1.data(), buf1.size());

  // For every Aline
  for (auto col_i = 0; col_i < rf.cols(); col_i++) {
    const Eigen::ArrayXd rf_aline = rf.col(col_i);
    std::span<const double> rf_aline_span(rf_aline.data(), rf_aline.size());

    Eigen::ArrayXd out_line = output.col(col_i);
    std::span<double> out_line_span(out_line.data(), out_line.size());

    fir_filter.forward(rf_aline_span, buf1span);
    envelope_detection.forward(buf1span, out_line_span);
  }
};

} // namespace arpam::recon