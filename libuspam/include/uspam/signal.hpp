#pragma once

#include <armadillo>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <uspam/fft.hpp>

namespace uspam::signal {

template <Floating T> void create_hamming_window(const std::span<T> window) {
  const auto numtaps = window.size();
  const auto N = static_cast<T>(numtaps - 1);

  // Hamming alpha = 0.54 or 25/46
  constexpr T ALPHA = 0.54;

  for (int i = 0; i < numtaps; ++i) {
    window[i] =
        ALPHA - (1 - ALPHA) * std::cos(2 * std::numbers::pi_v<T> * i / N);
  }
}

template <Floating T>
[[nodiscard]] auto create_hamming_window(const int numtaps) -> arma::Col<T> {
  arma::Col<T> window(numtaps);
  create_hamming_window<T>(window);
  return window;
}

// 1D linear interpolation for monotonically increasing sample points
// Conditions: xp must be increasing
template <Floating T>
void interp(std::span<const T> x, std::span<const T> xp, std::span<const T> fp,
            std::span<T> fx) {
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
      const auto denom = (xp[lower] - xp[lower - 1]);
      if (denom == 0) {
        fx[i] = fp[lower - 1];
      } else {
        const auto t = (x[i] - xp[lower - 1]) / denom;
        fx[i] = fp[lower - 1] + t * (fp[lower] - fp[lower - 1]);
      }
    }
  }
}

// 1D linear interpolation for monotonically increasing sample points
// Conditions: xp must be increasing

template <Floating T>
[[nodiscard]] auto interp(const arma::Col<T> &x, const arma::Col<T> &xp,
                          const arma::Col<T> &fp) {
  arma::Col<T> fx(x.size());
  interp(x, xp, fp, fx);
  return fx;
}
/**
@brief FIR filter design using the window method.

Ported from `scipy.signal`
From the given frequencies `freq` and corresponding gains `gain`,
this function constructs an FIR filter with linear phase and
(approximately) the given frequency response. A Hamming window will be applied
to the result

@param numtaps The number of taps in the FIR filter. `numtaps` must be less than
`nfreqs`.
@param freq The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
Nyquist. The Nyquist is half `fs`. The values in `freq` must be nondescending.
A value can be repeated once to implement a discontinuity. The first value in
`freq` must be 0, and the last value must be ``fs/``. Values 0 and ``fs/2`` must
not be repeated
@param gain The filter gains at the frequency sampling points. Certain
constraints to gain values, depending on filter type, are applied.
@param nfreqs The size of the interpolation mesh used to construct the filter.
For most efficient behavior, this should be a power of 2 plus 1 (e.g. 129, 257).
The default is one more than the smallest power of 2 that is not less that
`numtaps`. `nfreqs` must be greater than `numtaps`
@param fs The sampling frequency of the signal. Each frequency in `cutoff` must
be between 0 and ``fs/2``. Default is 2.

@return Eigen::ArrayXd The filter coefficients of the FIR filter, as a 1-D array
of length numtaps
*/
template <Floating T>
auto firwin2(int numtaps, const std::span<const T> freq,
             const std::span<const T> gain, int nfreqs = 0,
             T fs = 2) -> arma::Col<T> {
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

  const T nyq = 0.5 * fs;
  if (nfreqs == 0) {
    nfreqs = 1 + static_cast<int>(std::pow(2, std::ceil(std::log2(numtaps))));
  }

  using CxVec = arma::Col<std::complex<T>>;

  // Linearly interpolate the desired response on a uniform mesh `x`
  const arma::Col<T> x = arma::linspace<arma::vec>(0, nyq, nfreqs);
  CxVec fx;
  {
    arma::Col<T> _fx(x.size(), arma::fill::none);
    interp<T>(x, freq, gain, _fx);
    fx = arma::conv_to<CxVec>::from(_fx);
  }

  // Adjust phase of the coefficients so that the first `ntaps` of the
  // inverse FFT are the desired filter coefficients
  CxVec shift =
      arma::exp(-static_cast<T>(numtaps - 1) / 2 * std::complex<T>(0, 1) *
                std::numbers::pi_v<T> * x / nyq);
  fx %= shift;

  // Compute the inverse fft
  const int real_size = (static_cast<int>(fx.size()) - 1) * 2;

  auto &fft_engine = fft::engine_c2r_1d<T>::get(real_size);
  // NOLINTBEGIN(*-pointer-arithmetic)
  for (int i = 0; i < nfreqs; ++i) {
    fft_engine.complex[i][0] = fx.at(i).real();
    fft_engine.complex[i][1] = fx.at(i).imag();
  }
  // NOLINTEND(*-pointer-arithmetic)
  fft_engine.execute();

  // Keep only the first `numtaps` coefficients (and normalize since FFTW
  // doesn't) and apply the Hamming window
  arma::Col<T> out = create_hamming_window<T>(numtaps);
  const auto fct = static_cast<T>(1. / real_size);
  for (int i = 0; i < numtaps; ++i) {
    out[i] *= fft_engine.real[i] * fct; // NOLINT
  }

  return out;
}

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <Floating T>
void hilbert_abs(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  auto &engine = fft::engine_1d<T>::get(n);

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
  const auto fct = 1. / static_cast<T>(n);
  for (auto i = 0; i < n; ++i) {
    const auto real = x[i];
    const auto imag = engine.in[i][1] * fct;
    env[i] = std::abs(std::complex{real, imag});
  }
  // NOLINTEND(*-pointer-arithmetic, *-magic-numbers)
}

template <Floating T>
[[nodiscard]] inline auto hilbert_abs(const std::span<const T> x) {
  arma::Col<T> env(x.size());
  hilbert_abs<T>(x, env);
  return env;
}

/**
@brief Compute the analytic signal, using the Hilbert transform.
Optimized internally to use r2c transforms.
*/

template <Floating T>
void hilbert_abs_r2c(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  auto &engine = fft::engine_half_cx_1d<T>::get(n);

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
    const std::complex<T> fct{0, -1};
    for (auto &v : engine.complex) {
      std::complex<T> cx{v[0], v[1]};
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
    const auto fct = 1. / static_cast<T>(n);
    // NOLINTBEGIN(*-pointer-arithmetic)
    for (auto i = 0; i < n; ++i) {
      const T real = x[i];
      const T imag = engine.real[i] * fct;
      // env[i] = std::abs(std::complex<T>{real, imag});
      env[i] = std::sqrt(real * real + imag * imag); // Clang can vectorize this
    }
    // NOLINTEND(*-pointer-arithmetic)
  }
}

} // namespace uspam::signal
