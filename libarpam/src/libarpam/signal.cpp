#include <cassert>
#include <cmath>
#include <stdexcept>

#include "libarpam/signal.hpp"

namespace arpam::signal {

void create_hamming_window(const std::span<double> window) {
  const auto numtaps = window.size();
  for (int i = 0; i < numtaps; ++i) {
    // NOLINTBEGIN(*-magic-numbers)
    window[i] =
        0.54 - 0.46 * std::cos(2 * M_PI * i / static_cast<double>(numtaps - 1));
    // NOLINTEND(*-magic-numbers)
  }
}

void interp(std::span<const double> x, std::span<const double> xp,
            std::span<const double> fp, std::span<double> fx) {
  if (xp.size() != fp.size() || xp.size() < 2) {
    throw std::invalid_argument(
        "xp and fp must have the same size and at least two elements");
  }

  for (int i = 0; i < x.size(); ++i) {
    // Simple linear interpolation (replace )
    const auto lower =
        std::lower_bound(xp.begin(), xp.end(), x[i]) - xp.begin();
    if (lower == 0) {
      fx[i] = fp[0];
    } else if (lower >= xp.size()) {
      fx[i] = fp[fp.size() - 1];
    } else {
      const double denom = (xp[lower] - xp[lower - 1]);
      if (denom == 0) {
        fx[i] = fp[lower - 1];
      } else {
        const double t = (x[i] - xp[lower - 1]) / denom;
        fx[i] = fp[lower - 1] + t * (fp[lower] - fp[lower - 1]);
      }
    }
  }
}

auto interp(const Eigen::ArrayXd &x, const Eigen::ArrayXd &xp,
            const Eigen::ArrayXd &fp) {
  Eigen::ArrayXd fx(x.size());
  interp(std::span<const double>(x.data(), x.size()),
         std::span<const double>(xp.data(), xp.size()),
         std::span<const double>(fp.data(), fp.size()),
         std::span<double>(fx.data(), fx.size()));
  return fx;
}

auto firwin2(int numtaps, const Eigen::ArrayXd &freq,
             const Eigen::ArrayXd &gain, int nfreqs, double fs)
    -> Eigen::ArrayXd {
  if (numtaps < 3 || numtaps % 2 == 0) {
    throw std::invalid_argument(
        "numtaps must be odd and greater or equal to 3.");
  }

  const double nyq = 0.5 * fs;
  if (nfreqs == 0) {
    nfreqs = 1 + static_cast<int>(std::pow(2, std::ceil(std::log2(numtaps))));
  }

  // Linearly interpolate the desired response on a uniform mesh `x`
  const Eigen::ArrayXd x = Eigen::ArrayXd::LinSpaced(nfreqs, 0.0, nyq);
  Eigen::ArrayXcd fx = interp(x, freq, gain);

  // Adjust phase of the coefficients so that the first `ntaps` of the
  // inverse FFT are the desired filter coefficients
  Eigen::ArrayXcd shift =
      Eigen::exp(-static_cast<double>(numtaps - 1) / 2 *
                 std::complex<double>(0, 1) * M_PI * x.array() / nyq);
  fx *= shift;

  // Compute the inverse fft
  const int real_size = (static_cast<int>(fx.size()) - 1) * 2;

  fft::fftw_engine_c2r_1d fft_engine(real_size);
  // NOLINTBEGIN(*-pointer-arithmetic)
  for (int i = 0; i < nfreqs; ++i) {
    fft_engine.complex[i][0] = fx(i).real();
    fft_engine.complex[i][1] = fx(i).imag();
  }
  // NOLINTEND(*-pointer-arithmetic)
  fft_engine.execute();

  // Keep only the first `numtaps` coefficients (and normalize since FFTW
  // doesn't) and apply the Hamming window
  Eigen::ArrayXd out = create_hamming_window(numtaps);
  for (int i = 0; i < numtaps; ++i) {
    out(i) *= fft_engine.real[i] / real_size; // NOLINT
  }

  return out;
}

auto hilbert(const Eigen::ArrayXd &input) -> Eigen::ArrayXcd {
  const Eigen::Index n = input.size();
  fft::fftw_engine_1d engine(n);

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
  // NOLINTBEGIN(*-pointer-arithmetic)
  for (Eigen::Index i = 0; i < n; ++i) {
    analytic_signal(i) = std::complex<double>(
        input[i], engine.in[i][1] / static_cast<double>(n));
  }
  // NOLINTEND(*-pointer-arithmetic)

  return analytic_signal;
}

void EnvelopDetection::operator()(const Eigen::ArrayXd &input,
                                  Eigen::ArrayXd &env_out) {
  assert(input.size() == env_out.size());
  const Eigen::Index n = input.size();

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
    env_out(i) = std::abs(std::complex{real, imag});
  }
}

} // namespace arpam::signal