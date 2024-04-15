#pragma once

#include <span>

#include <armadillo>
#include <fftw3.h>

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
auto firwin2(int numtaps, const std::span<const double> freq,
             const std::span<const double> gain, int nfreqs = 0, double fs = 2)
    -> arma::vec;

/**
@brief Compute the analytic signal, using the Hilbert transform.
*/
void hilbert_abs(const std::span<const double> x, const std::span<double> env);
[[nodiscard]] inline auto hilbert_abs(const std::span<const double> x) {
  arma::vec env(x.size());
  hilbert_abs(x, env);
  return env;
}


void hilbert_abs_r2c(const std::span<const double> x,
                     const std::span<double> env);

} // namespace uspam::signal
