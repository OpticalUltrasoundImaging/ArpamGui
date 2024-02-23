#pragma once

#include "fftconv.hpp"
#include <Eigen/Dense>
#include <fftw3.h>

#include "libarpam/io.hpp"
#include "libarpam/signal.hpp"

namespace arpam::recon {

/*
Filt
*/
template <typename T> void recon(Eigen::MatrixX<T> rf) {
  for (auto col_i = 0; col_i < rf.cols(); col_i++) {
    auto rf_aline = rf.col(col_i);
  }
};

} // namespace arpam::recon
