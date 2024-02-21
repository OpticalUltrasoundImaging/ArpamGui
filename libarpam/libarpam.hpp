#pragma once

#include <span>

#include "fftconv.hpp"

namespace arpam {

namespace recon {

//void apply_fir_filt(const double *x, size_t x_size, const double *kernel,
                    //size_t kernel_size, double *res, size_t res_size) {
  //fftconv::oaconvolve_fftw(x, x_size, kernel, kernel_size, res, res_size);
//}

} // namespace recon

/*
Filt
*/
void recon_ascan();
} // namespace arpam
