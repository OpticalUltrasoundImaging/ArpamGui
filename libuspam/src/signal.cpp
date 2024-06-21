#include <cmath>
#include <numbers>
#include <stdexcept>

#include "uspam/fft.hpp"
#include "uspam/signal.hpp"

namespace uspam::signal {

void create_hamming_window(const std::span<double> window) {
  const auto numtaps = window.size();
  const auto N = static_cast<double>(numtaps - 1);

  // Hamming alpha = 0.54 or 25/46
  constexpr double ALPHA = 0.54;

  for (int i = 0; i < numtaps; ++i) {
    window[i] = ALPHA - (1 - ALPHA) * std::cos(2 * std::numbers::pi * i / N);
  }
}

auto create_hamming_window(const int numtaps) -> arma::vec {
  arma::vec window(numtaps);
  create_hamming_window(window);
  return window;
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

auto interp(const arma::vec &x, const arma::vec &xp, const arma::vec &fp) {
  arma::vec fx(x.size());
  interp(x, xp, fp, fx);
  return fx;
}

auto firwin2(int numtaps, const std::span<const double> freq,
             const std::span<const double> gain, int nfreqs, double fs)
    -> arma::vec {
  if (numtaps < 3 || numtaps % 2 == 0) {
    throw std::invalid_argument(
        "numtaps must be odd and greater or equal to 3.");
  }

  if (freq.size() != gain.size()) {
    throw std::invalid_argument("freq and gain must have the same size.\n");
  }
  if (freq[0] != 0) {
    throw std::invalid_argument("freq[0] must be 0\n");
  }

  const double nyq = 0.5 * fs;
  if (nfreqs == 0) {
    nfreqs = 1 + static_cast<int>(std::pow(2, std::ceil(std::log2(numtaps))));
  }

  // Linearly interpolate the desired response on a uniform mesh `x`
  const arma::vec x = arma::linspace<arma::vec>(0, nyq, nfreqs);
  arma::cx_vec fx;
  {
    arma::vec _fx(x.size(), arma::fill::none);
    interp(x, freq, gain, _fx);
    fx = arma::conv_to<arma::cx_vec>::from(_fx);
  }

  // Adjust phase of the coefficients so that the first `ntaps` of the
  // inverse FFT are the desired filter coefficients
  arma::cx_vec shift =
      arma::exp(-static_cast<double>(numtaps - 1) / 2 *
                std::complex<double>(0, 1) * std::numbers::pi * x / nyq);
  fx %= shift;

  // Compute the inverse fft
  const int real_size = (static_cast<int>(fx.size()) - 1) * 2;

  auto &fft_engine = fft::engine_c2r_1d<double>::get(real_size);
  // NOLINTBEGIN(*-pointer-arithmetic)
  for (int i = 0; i < nfreqs; ++i) {
    fft_engine.complex[i][0] = fx(i).real();
    fft_engine.complex[i][1] = fx(i).imag();
  }
  // NOLINTEND(*-pointer-arithmetic)
  fft_engine.execute();

  // Keep only the first `numtaps` coefficients (and normalize since FFTW
  // doesn't) and apply the Hamming window
  arma::vec out = create_hamming_window(numtaps);
  const double fct = 1. / real_size;
  for (int i = 0; i < numtaps; ++i) {
    out(i) *= fft_engine.real[i] * fct; // NOLINT
  }

  return out;
}

void hilbert_abs(const std::span<const double> x, const std::span<double> env) {
  const auto n = x.size();
  auto &engine = fft::fftw_engine_1d<double>::get(n);

  // Copy input to real buffer
  // NOLINTBEGIN(*-pointer-arithmetic, *-magic-numbers)
  for (int i = 0; i < n; ++i) {
    engine.in[i][0] = x[i];
    engine.in[i][1] = 0.;
  }

  // Execute r2c fft
  engine.execute_forward();

  // Zero negative frequencies (half-Hermitian to Hermitian conversion)
  // Double the magnitude of positive frequencies
  const auto n_half = n / 2;
  for (auto i = 1; i < n_half; ++i) {
    engine.out[i][0] *= 2.;
    engine.out[i][1] *= 2.;
  }

  if (n % 2 == 0) {
    engine.out[n_half][0] = 0.;
    engine.out[n_half][1] = 0.;
  } else {
    engine.out[n_half][0] *= 2.;
    engine.out[n_half][1] *= 2.;
  }

  for (auto i = n_half + 1; i < n; ++i) {
    engine.out[i][0] = 0.;
    engine.out[i][1] = 0.;
  }

  // Execute c2r fft on modified spectrum
  engine.execute_backward();

  // Construct the analytic signal
  const double fct = 1. / static_cast<double>(n);
  for (auto i = 0; i < n; ++i) {
    const auto real = x[i];
    const auto imag = engine.in[i][1] * fct;
    env[i] = std::abs(std::complex{real, imag});
  }
  // NOLINTEND(*-pointer-arithmetic, *-magic-numbers)
}

void hilbert_abs_r2c(const std::span<const double> x,
                     const std::span<double> env) {
  const auto n = x.size();
  auto &engine = fft::fftw_engine_half_cx_1d<double>::get(n);

  // Copy input to real buffer
  // NOLINTBEGIN(*-pointer-arithmetic)
  for (int i = 0; i < n; ++i) {
    engine.real[i] = x[i];
  }
  // NOLINTEND(*-pointer-arithmetic)

  // Execute r2c fft
  engine.execute_r2c();

  // NOLINTBEGIN(*-pointer-arithmetic)
  // Only positive frequencies in r2c transform. Switch freq by *-1j
  {
    const std::complex<double> fct{0, -1};
    for (auto &v : engine.complex) {
      std::complex<double> cx{v[0], v[1]};
      cx *= fct;
      v[0] = cx.real();
      v[1] = cx.imag();
    }
  }
  // NOLINTEND(*-pointer-arithmetic)

  // Execute c2r fft on modified spectrum
  engine.execute_c2r();

  {
    // Construct the analytic signal
    const double fct = 1. / static_cast<double>(n);
    // NOLINTBEGIN(*-pointer-arithmetic)
    for (auto i = 0; i < n; ++i) {
      const auto real = x[i];
      const auto imag = engine.real[i] * fct;
      env[i] = std::abs(std::complex{real, imag});
    }
    // NOLINTEND(*-pointer-arithmetic)
  }
}
} // namespace uspam::signal
