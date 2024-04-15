#pragma once
#include <cstdio>

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    auto status = static_cast<cudaError_t>(call);                              \
    if (status != cudaSuccess)                                                 \
      fprintf(stderr,                                                          \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "       \
              "with "                                                          \
              "%s (%d).\n",                                                    \
              #call, __LINE__, __FILE__, cudaGetErrorString(status), status);  \
  }
#endif // CUDA_RT_CALL

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                       \
  {                                                                            \
    auto status = static_cast<cufftResult>(call);                              \
    if (status != CUFFT_SUCCESS)                                               \
      fprintf(stderr,                                                          \
              "ERROR: CUFFT call \"%s\" in line %d of file %s failed "         \
              "with "                                                          \
              "code (%d).\n",                                                  \
              #call, __LINE__, __FILE__, status);                              \
  }
#endif // CUFFT_CALL

