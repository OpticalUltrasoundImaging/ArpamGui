#pragma once

#include <armadillo>
#include <cmath>
#include <fftconv/fftconv.hpp>
#include <fftconv/fftw.hpp>
#include <numbers>
#include <span>
#include <stdexcept>

namespace uspam::signal {

template <fftw::Floating T>
void create_hamming_window(const std::span<T> window) {
  const auto numtaps = window.size();
  const auto N = static_cast<T>(numtaps - 1);

  // Hamming alpha = 0.54 or 25/46
  constexpr T ALPHA = 0.54;

  for (int i = 0; i < numtaps; ++i) {
    window[i] =
        ALPHA - (1 - ALPHA) * std::cos(2 * std::numbers::pi_v<T> * i / N);
  }
}

template <fftw::Floating T>
[[nodiscard]] auto create_hamming_window(const int numtaps) -> arma::Col<T> {
  arma::Col<T> window(numtaps);
  create_hamming_window<T>(window);
  return window;
}

// 1D linear interpolation for monotonically increasing sample points
// Conditions: xp must be increasing
template <fftw::Floating T>
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

template <fftw::Floating T>
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
template <fftw::Floating T>
auto firwin2(int numtaps, const std::span<const T> freq,
             const std::span<const T> gain, int nfreqs = 0, T fs = 2)
    -> arma::Col<T> {
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

  fftw::EngineR2C1D<T> &engine = fftw::EngineR2C1D<T>::get(real_size);
  // NOLINTBEGIN(*-pointer-arithmetic)
  for (int i = 0; i < nfreqs; ++i) {
    engine.buf.out[i][0] = fx.at(i).real();
    engine.buf.out[i][1] = fx.at(i).imag();
  }
  // NOLINTEND(*-pointer-arithmetic)
  engine.backward();

  // Keep only the first `numtaps` coefficients (and normalize since FFTW
  // doesn't) and apply the Hamming window
  arma::Col<T> out = create_hamming_window<T>(numtaps);
  const auto fct = static_cast<T>(1. / real_size);
  for (int i = 0; i < numtaps; ++i) {
    out[i] *= engine.buf.in[i] * fct; // NOLINT
  }

  return out;
}

} // namespace uspam::signal
