#pragma once

#include <mutex>
#include <span>
#include <unordered_map>
#include <iostream>

#include <fftw3.h>

namespace uspam::fft {

inline static std::mutex *_fftw_mutex;
inline void use_fftw_mutex(std::mutex *fftw_mutex) {
    if (!fftw_mutex){
        std::cerr << "Warning: passed a nullptr to uspam::fft::use_fftw_mutex!\n";
    }
    _fftw_mutex = fftw_mutex;
}

inline std::mutex* get_fftw_mutex() {
    return _fftw_mutex;
}

// In memory cache with key type K and value type V
// additionally accepts a mutex to guard the V constructor
template <class Key, class Val>
auto get_cached_vlock(Key key, std::mutex *V_mutex) {
  static thread_local std::unordered_map<Key, std::unique_ptr<Val>> _cache;

  auto &val = _cache[key];
  if (val == nullptr) {
    // Using unique_lock here for RAII locking since the mutex is optional.
    // If we have a non-optional mutex, prefer scoped_lock
    const auto lock = V_mutex == nullptr ? std::unique_lock<std::mutex>{}
                                         : std::unique_lock(*V_mutex);

    val = std::make_unique<Val>(key);
  }
  return val.get();
}

struct fftw_engine_r2c_1d {
  size_t n;
  double *real;
  fftw_complex *complex;
  fftw_plan plan;

  static auto get(size_t n) -> auto & {
    thread_local static auto cache =
        get_cached_vlock<size_t, fftw_engine_r2c_1d>;
    return *cache(n, get_fftw_mutex());
  }

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

  static auto get(size_t n) -> auto & {
    thread_local static auto cache =
        get_cached_vlock<size_t, fftw_engine_c2r_1d>;
    return *cache(n, get_fftw_mutex());
  }

  explicit fftw_engine_c2r_1d(size_t n)
      : real(fftw_alloc_real(n), n),
        complex(fftw_alloc_complex(n / 2 + 1), n / 2 + 1),
        plan(fftw_plan_dft_c2r_1d(static_cast<int>(n), complex.data(),
                                  real.data(), FFTW_ESTIMATE)) {
    {
        if (plan == nullptr) {
            int n = 0;
        }
    }
  }
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

  static auto get(size_t n) -> auto & {
    thread_local static auto cache =
        get_cached_vlock<size_t, fftw_engine_half_cx_1d>;
    return *cache(n, get_fftw_mutex());
  }

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

  static auto get(size_t n) -> auto & {
    thread_local static auto cache = get_cached_vlock<size_t, fftw_engine_1d>;
    return *cache(n, get_fftw_mutex());
  }

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

} // namespace uspam::fft
