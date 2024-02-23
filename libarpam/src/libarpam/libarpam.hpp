#pragma once

#include <cmath>
#include <stdexcept>

#include "fftconv.hpp"
#include <Eigen/Dense>
#include <fftw3.h>

#include "libarpam/io.hpp"

namespace arpam {

inline constexpr double PI = 3.1415926535897932384626;

namespace signal {

// Helper function to create a Hamming window
inline auto create_hamming_window(int numtaps) {
  Eigen::VectorXd window(numtaps);
  for (int i = 0; i < numtaps; ++i) {
    // NOLINTBEGIN(*-magic-numbers)
    window(i) = 0.54 - 0.46 * cos(2 * M_PI * i / (numtaps - 1));
    // NOLINTEND(*-magic-numbers)
  }
  return window;
}

// 1D linear interpolation for monotonically increasing sample points
// Conditions: xp must be increasing
inline auto interp(const Eigen::ArrayXd &x, const Eigen::ArrayXd &xp,
                   const Eigen::ArrayXd &fp) {
  if (xp.size() != fp.size() || xp.size() < 2) {
    throw std::invalid_argument(
        "xp and fp must have the same size and at least two elements");
  }

  Eigen::ArrayXd fx(x.size());
  for (int i = 0; i < x.size(); ++i) {
    // Simple linear interpolation (replace )
    const auto lower =
        std::lower_bound(xp.cbegin(), xp.cend(), x(i)) - xp.cbegin();
    if (lower == 0) {
      fx(i) = fp(0);
    } else if (lower >= xp.size()) {
      fx(i) = fp(fp.size() - 1);
    } else {
      const double denom = (xp(lower) - xp(lower - 1));
      if (denom == 0) {
        fx(i) = fp(lower - 1);
      } else {
        const double t = (x(i) - xp(lower - 1)) / denom;
        fx(i) = fp(lower - 1) + t * (fp(lower) - fp(lower - 1));
      }
    }
  }
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
inline auto firwin2(int numtaps, const Eigen::ArrayXd &freq,
                    const Eigen::ArrayXd &gain, int nfreqs = 0, double fs = 2.0)
    -> Eigen::ArrayXd {
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
      Eigen::exp(-static_cast<double>(numtaps - 1) / 2.0 *
                 std::complex<double>(0, 1) * M_PI * x.array() / nyq);
  fx *= shift;

  // Use compute the inverse fft
  const int real_size = (static_cast<int>(fx.size()) - 1) * 2;
  auto *real_buf = fftw_alloc_real(real_size);
  auto *cx_buf = fftw_alloc_complex(fx.size());
  for (int i = 0; i < nfreqs; ++i) {
    cx_buf[i][0] = fx(i).real();
    cx_buf[i][1] = fx(i).imag();
  }

  fftw_plan plan =
      fftw_plan_dft_c2r_1d(real_size, cx_buf, real_buf, FFTW_ESTIMATE);

  fftw_execute(plan);

  // Keep only the first `numtaps` coefficients (and normalize since FFTW
  // doesn't) and apply the Hamming window
  Eigen::ArrayXd out = create_hamming_window(numtaps);
  for (int i = 0; i < numtaps; ++i) {
    out(i) *= real_buf[i] / real_size; // NOLINT
  }

  fftw_destroy_plan(plan);
  fftw_free(real_buf);
  fftw_free(cx_buf);

  return out;
}
} // namespace signal

namespace recon {

/*
Filt
*/
template <typename T> void recon(Eigen::MatrixX<T> rf) {
  for (auto col_i = 0; col_i < rf.cols(); col_i++) {
    auto rf_aline = rf.col(col_i);
  }
};

} // namespace recon

} // namespace arpam
