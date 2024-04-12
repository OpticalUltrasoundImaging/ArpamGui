#pragma once

#include "uspam/cudaUtil.cuh"

#include <cuda_runtime.h>
#include <cufft.h>


void hilbertTransform(const float *input, float *output, int N);
void hilbert2(const double *x, double *out, const int rows, const int cols);
