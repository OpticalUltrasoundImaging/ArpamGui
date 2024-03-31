#include "libarpam/recon.hpp"

namespace arpam::recon {

void recon(const ReconParams<double> &params, const Eigen::MatrixXd &rf,
           Eigen::MatrixXd &output) {

  if (rf.cols() != output.cols() || rf.rows() != output.rows()) {
    throw std::invalid_argument("Sizes of rf and output are different.");
  }

  const auto aline_size = rf.rows();
  //   const auto num_alines = rf.cols();

  pipeline::FIRFilter firFilter(aline_size, params.firFilterParams);
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

  pipeline::FIRFilter firFilter(aline_size, params.firFilterParams);
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

void pipeline::EnvelopeDetection::forward(const std::span<const double> input,
                                          std::span<double> output) {
  assert(input.size() == output.size());
  const auto n = static_cast<int64_t>(input.size());

  // Copy input to real buffer
  // NOLINTBEGIN(*-pointer-arithmetic)
  for (int i = 0; i < n; ++i) {
    engine.in[i][0] = input[i];
    engine.in[i][1] = 0.;
  }
  // NOLINTEND(*-pointer-arithmetic)

  // Execute r2c fft
  engine.execute_forward();

  // NOLINTBEGIN(*-pointer-arithmetic)
  // Zero negative frequencies (half-Hermitian to Hermitian conversion)
  // Double the magnitude of positive frequencies
  const auto n_half = n / 2;
  for (auto i = 1; i < n_half; ++i) {
    engine.out[i][0] *= 2;
    engine.out[i][1] *= 2;
  }

  if (n % 2 == 0) {
    engine.out[n_half][0] = 0;
    engine.out[n_half][1] = 0;
  } else {
    engine.out[n_half][0] *= 2;
    engine.out[n_half][1] *= 2;
  }

  for (auto i = n_half + 1; i < n; ++i) {
    engine.out[i][0] = 0;
    engine.out[i][1] = 0;
  }
  // NOLINTEND(*-pointer-arithmetic)

  // Execute c2r fft on modified spectrum
  engine.execute_backward();

  // Construct the analytic signal
  Eigen::ArrayXcd analytic_signal(n);
  const double fct = 1. / static_cast<double>(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto real = input[i];
    const auto imag = engine.in[i][1] * fct;
    output[i] = std::abs(std::complex{real, imag});
  }
};

void pipeline::FIRFilter::forward(const std::span<const T> input,
                                  const std::span<T> output) const {
  // Convolution with kernel using "same" padding
  assert(input.size() <= output.size());
  fftconv::oaconvolve_fftw_same<T>(
      input, std::span<const T>(m_kernel.data(), m_kernel.size()),
      std::span<T>(output.data(), input.size()));
}

void pipeline::FIRFilter::forward(const Eigen::MatrixX<T> &input,
                                  Eigen::MatrixX<T> &output) const {
  for (auto col_i = 0; col_i < input.cols(); col_i++) {
    const auto in_col = input.col(col_i);
    auto out_col = output.col(col_i);
    this->forward(std::span<const T>(in_col.data(), in_col.size()),
                  std::span<T>(out_col.data(), out_col.size()));
  }
}

void pipeline::FIRFilter::setKernel(const FIRFilterParams<T> &params) {
  if (!params.validate()) {
    throw std::runtime_error("Bad FIRFilterParams. Check stderr.");
  }
  m_kernel = signal::firwin2(params.numtaps, params.freq, params.gain);
}

void pipeline::LogCompression::forward(const std::span<const double> input,
                                       std::span<double> output) const {
  const auto max_val = *std::max_element(input.begin(), input.end());
  if (max_val != 0) {
    const auto scale = 1 / max_val;
    for (int i = 0; i < input.size(); ++i) {
      const double compressed =
          20.0 / m_db * (std::log10(input[i] * scale) + 1);
      output[i] = compressed < 0 ? 0 : compressed;
    }
  } else {
    std::fill(output.begin(), output.end(), 0);
  }
}

void pipeline::LogCompression::forward(const std::span<const double> input,
                                       std::span<uint8_t> output) const {
  const auto max_val = *std::max_element(input.begin(), input.end());
  if (max_val != 0) {
    const auto scale = 1 / max_val;
    for (int i = 0; i < input.size(); ++i) {
      const auto compressed = static_cast<uint8_t>(
          255.0 * (20.0 / m_db * (std::log10(input[i] * scale) + 1)));
      output[i] = compressed < 1 ? 0 : compressed;
    }
  } else {
    std::fill(output.begin(), output.end(), 0);
  }
}
} // namespace arpam::recon