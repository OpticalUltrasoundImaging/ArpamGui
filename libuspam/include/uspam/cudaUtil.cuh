#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

// Handle CUDA errors
inline void handleCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in " << file << " at line " << line << ": "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))

// Handle cuFFT errors
inline void handleCufftError(cufftResult err, const char *file, int line) {
  if (err != CUFFT_SUCCESS) {
    std::cerr << "cuFFT error in " << file << " at line " << line << ": " << err
              << std::endl;
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_CUFFT_ERROR(err) (handleCufftError(err, __FILE__, __LINE__))
