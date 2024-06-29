#pragma once

#include <armadillo>
#include <span>
#include <uspam/fft.hpp>

namespace uspam::signal {

void create_hamming_window(std::span<double> window);

// Helper function to create a Hamming window
[[nodiscard]] auto create_hamming_window(int numtaps) -> arma::vec;

// 1D linear interpolation for monotonically increasing sample points
// Conditions: xp must be increasing
void interp(std::span<const double> x, std::span<const double> xp,
            std::span<const double> fp, std::span<double> fx);

// 1D linear interpolation for monotonically increasing sample points
// Conditions: xp must be increasing
[[nodiscard]] auto interp(const arma::vec &x, const arma::vec &xp,
                          const arma::vec &fp);

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
auto firwin2(int numtaps, std::span<const double> freq,
             std::span<const double> gain, int nfreqs = 0, double fs = 2)
    -> arma::vec;

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
template <Floating T>
void hilbert_abs(const std::span<const T> x, const std::span<T> env) {
  const auto n = x.size();
  auto &engine = fft::fftw_engine_1d<T>::get(n);

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
  auto &engine = fft::fftw_engine_half_cx_1d<T>::get(n);

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
      env[i] = std::abs(std::complex<T>{real, imag});
    }
    // NOLINTEND(*-pointer-arithmetic)
  }
}
} // namespace uspam::signal
