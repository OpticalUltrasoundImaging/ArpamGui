#include "libarpam/recon.hpp"

namespace arpam::recon {

void recon(const ReconParams<double> &params, const Eigen::MatrixXd &rf,
           Eigen::MatrixXd &output) {

  if (rf.cols() != output.cols() || rf.rows() != output.rows()) {
    throw std::invalid_argument("Sizes of rf and output are different.");
  }

  const auto aline_size = rf.rows();
  //   const auto num_alines = rf.cols();

  pipeline::FIRFilter<double> firFilter(aline_size, params.firFilterParams);
  pipeline::EnvelopeDetection envelopeDetection(aline_size);
  pipeline::LogCompression logCompress(aline_size, params.dynamic_range);

  // For every Aline
  for (auto col_i = 0; col_i < rf.cols(); col_i++) {
    auto rf_aline = rf.col(col_i);
    std::span<const double> rf_aline_span(rf_aline.data(), rf_aline.size());

    auto out_line = output.col(col_i);
    std::span<double> out_line_span(out_line.data(), out_line.size());

    firFilter.forward(rf_aline_span, out_line_span);
    envelopeDetection.forward(out_line_span, out_line_span);
    logCompress.forward(out_line_span, out_line_span);
  }
};

void recon(const ReconParams<double> &params, const Eigen::MatrixXd &rf,
           Eigen::MatrixX<uint8_t> &output) {

  if (rf.cols() != output.cols() || rf.rows() != output.rows()) {
    throw std::invalid_argument("Sizes of rf and output are different.");
  }

  const auto aline_size = rf.rows();
  //   const auto num_alines = rf.cols();

  pipeline::FIRFilter<double> firFilter(aline_size, params.firFilterParams);
  pipeline::EnvelopeDetection envelopeDetection(aline_size);
  pipeline::LogCompression logCompress(aline_size, params.dynamic_range);

  std::vector<double> tmpbuf(aline_size);

  // For every Aline
  for (auto col_i = 0; col_i < rf.cols(); col_i++) {
    auto rf_aline = rf.col(col_i);
    std::span<const double> rf_aline_span(rf_aline.data(), rf_aline.size());

    auto out_line = output.col(col_i);
    std::span out_line_span(out_line.data(), out_line.size());

    firFilter.forward(rf_aline_span, tmpbuf);
    envelopeDetection.forward(tmpbuf, tmpbuf);
    logCompress.forward(tmpbuf, out_line_span);
  }
};
} // namespace arpam::recon