#pragma once

#include "uspam/cudaUtil.cuh"

#include <cuda_runtime.h>
#include <cufft.h>


void hilbert2(const double *x, double *out, const int rows, const int cols);
