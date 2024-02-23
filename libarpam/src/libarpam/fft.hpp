#pragma once

#include <span>

#include <fftw3.h>

namespace arpam::fft {

struct fftw_engine_r2c_1d {
  size_t n;
  double *real;
  fftw_complex *complex;
  fftw_plan plan;

  explicit fftw_engine_r2c_1d(size_t n)
      : n(n), real(fftw_alloc_real(n)), complex(fftw_alloc_complex(n / 2 + 1)),
        plan(fftw_plan_dft_r2c_1d(static_cast<int>(n), real, complex,
                                  FFTW_ESTIMATE)) {}
  fftw_engine_r2c_1d(const fftw_engine_r2c_1d &) = delete;
  auto operator=(const fftw_engine_r2c_1d &) -> fftw_engine_r2c_1d & = delete;
  fftw_engine_r2c_1d(fftw_engine_r2c_1d &&) = delete;
  auto operator=(fftw_engine_r2c_1d &&) -> fftw_engine_r2c_1d & = delete;

  ~fftw_engine_r2c_1d() {
    fftw_destroy_plan(plan);
    fftw_free(real);
    fftw_free(complex);
  }

  inline void execute() const { fftw_execute(plan); }
};

struct fftw_engine_c2r_1d {
  std::span<double> real;
  std::span<fftw_complex> complex;
  fftw_plan plan;

  explicit fftw_engine_c2r_1d(size_t n)
      : real(fftw_alloc_real(n), n),
        complex(fftw_alloc_complex(n / 2 + 1), n / 2 + 1),
        plan(fftw_plan_dft_c2r_1d(static_cast<int>(n), complex.data(),
                                  real.data(), FFTW_ESTIMATE)) {}
  ~fftw_engine_c2r_1d() {
    fftw_destroy_plan(plan);
    fftw_free(real.data());
    fftw_free(complex.data());
  }
  fftw_engine_c2r_1d(const fftw_engine_c2r_1d &) = delete;
  auto operator=(const fftw_engine_c2r_1d &) -> fftw_engine_c2r_1d & = delete;
  fftw_engine_c2r_1d(fftw_engine_c2r_1d &&) = delete;
  auto operator=(fftw_engine_c2r_1d &&) -> fftw_engine_c2r_1d & = delete;

  inline void execute() const { fftw_execute(plan); }
};

struct fftw_engine_half_cx_1d {
  std::span<double> real;
  std::span<fftw_complex> complex;
  fftw_plan plan_f;
  fftw_plan plan_b;

  explicit fftw_engine_half_cx_1d(size_t n)
      : real(fftw_alloc_real(n), n),
        complex(fftw_alloc_complex(n / 2 + 1), n / 2 + 1),
        plan_f(fftw_plan_dft_r2c_1d(static_cast<int>(n), real.data(),
                                    complex.data(), FFTW_ESTIMATE)),
        plan_b(fftw_plan_dft_c2r_1d(static_cast<int>(n), complex.data(),
                                    real.data(), FFTW_ESTIMATE)) {}
  fftw_engine_half_cx_1d(const fftw_engine_half_cx_1d &) = default;
  auto operator=(const fftw_engine_half_cx_1d &)
      -> fftw_engine_half_cx_1d & = default;
  fftw_engine_half_cx_1d(fftw_engine_half_cx_1d &&) = delete;
  auto operator=(fftw_engine_half_cx_1d &&)
      -> fftw_engine_half_cx_1d & = delete;
  ~fftw_engine_half_cx_1d() {
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);
    fftw_free(real.data());
    fftw_free(complex.data());
  }

  inline void execute_r2c() const { fftw_execute(plan_f); }
  inline void execute_c2r() const { fftw_execute(plan_b); }
};

struct fftw_engine_1d {
  std::span<fftw_complex> in;
  std::span<fftw_complex> out;
  fftw_plan plan_f;
  fftw_plan plan_b;

  explicit fftw_engine_1d(size_t n)
      : in(fftw_alloc_complex(n), n), out(fftw_alloc_complex(n), n),
        plan_f(fftw_plan_dft_1d(static_cast<int>(n), in.data(), out.data(),
                                FFTW_FORWARD, FFTW_ESTIMATE)),
        plan_b(fftw_plan_dft_1d(static_cast<int>(n), out.data(), in.data(),
                                FFTW_BACKWARD, FFTW_ESTIMATE)) {}
  fftw_engine_1d(const fftw_engine_1d &) = default;
  auto operator=(const fftw_engine_1d &) -> fftw_engine_1d & = default;
  fftw_engine_1d(fftw_engine_1d &&) = delete;
  auto operator=(fftw_engine_1d &&) -> fftw_engine_1d & = delete;
  ~fftw_engine_1d() {
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_b);
    fftw_free(in.data());
    fftw_free(out.data());
  }

  inline void execute_forward() const { fftw_execute(plan_f); }
  inline void execute_backward() const { fftw_execute(plan_b); }
};

} // namespace arpam::fft