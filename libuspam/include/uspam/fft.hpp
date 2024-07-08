#pragma once

#include <complex>
#include <cstdlib>
#include <fftw3.h>
#include <span>
#include <type_traits>
#include <unordered_map>

namespace uspam {
template <typename T>
concept Floating = std::is_floating_point_v<T>;
}

namespace uspam::fftw {

namespace details {

template <Floating T> struct ComplexTraits {};
template <> struct ComplexTraits<double> {
  using Type = fftw_complex;
};
template <> struct ComplexTraits<float> {
  using Type = fftwf_complex;
};

template <Floating T> struct PlanTraits;
template <> struct PlanTraits<double> {
  using Type = fftw_plan;
};
template <> struct PlanTraits<float> {
  using Type = fftwf_plan;
};

} // namespace details

template <Floating T> struct FloatingTraits {
  using Real = T;
  using Complex = std::complex<T>;
};
template <Floating T> using Complex = details::ComplexTraits<T>::Type;
template <Floating T> using Plan = details::PlanTraits<T>::Type;

/**
fftw_alloc_real
 */
template <Floating T> inline T *alloc_real(size_t n);
template <> inline double *alloc_real<double>(size_t n) {
  return fftw_alloc_real(n);
};
template <> inline float *alloc_real<float>(size_t n) {
  return fftwf_alloc_real(n);
};

/**
fftw_alloc_complex
 */
template <Floating T> Complex<T> inline *alloc_complex(size_t n);
template <> inline Complex<double> *alloc_complex<double>(size_t n) {
  return fftw_alloc_complex(n);
};
template <> inline Complex<float> *alloc_complex<float>(size_t n) {
  return fftwf_alloc_complex(n);
};

/**
fftw_free
 */
template <Floating T> inline void free(void *ptr);
template <> inline void free<double>(void *ptr) { fftw_free(ptr); }
template <> inline void free<float>(void *ptr) { fftwf_free(ptr); }

/**
fftw_execute
 */
template <Floating T> inline void execute(Plan<T> plan);
template <> void inline execute<double>(Plan<double> plan) {
  fftw_execute(plan);
}
template <> void inline execute<float>(Plan<float> plan) {
  fftwf_execute(plan);
}

/**
fftw_destroy_plan
 */
template <Floating T> inline void destroy_plan(Plan<T> plan);
template <> inline void destroy_plan<double>(Plan<double> plan) {
  fftw_destroy_plan(plan);
}
template <> inline void destroy_plan<float>(Plan<float> plan) {
  fftwf_destroy_plan(plan);
}

/**
fftw_plan_dft_1d
 */
template <Floating T>
inline Plan<T> plan_dft_1d(int n, Complex<T> *in, Complex<T> *out, int sign,
                           unsigned int flags);

template <>
inline Plan<double> plan_dft_1d<double>(int n, Complex<double> *in,
                                        Complex<double> *out, int sign,
                                        unsigned int flags) {
  return fftw_plan_dft_1d(n, in, out, sign, flags);
}
template <>
inline Plan<float> plan_dft_1d<float>(int n, Complex<float> *in,
                                      Complex<float> *out, int sign,
                                      unsigned int flags) {
  return fftwf_plan_dft_1d(n, in, out, sign, flags);
}

/**
fftw_plan_dft_r2c_1d
 */
template <Floating T>
inline Plan<T> plan_dft_r2c_1d(int n, T *in, Complex<T> *out,
                               unsigned int flags);
template <>
inline Plan<double> plan_dft_r2c_1d<double>(int n, double *in,
                                            Complex<double> *out,
                                            unsigned int flags) {
  return fftw_plan_dft_r2c_1d(n, in, out, flags);
}
template <>
inline Plan<float> plan_dft_r2c_1d<float>(int n, float *in, Complex<float> *out,
                                          unsigned int flags) {
  return fftwf_plan_dft_r2c_1d(n, in, out, flags);
}

/**
fftw_plan_dft_c2r_1d
 */
template <Floating T>
inline Plan<T> plan_dft_c2r_1d(int n, Complex<T> *in, T *out,
                               unsigned int flags);
template <>
inline Plan<double> plan_dft_c2r_1d(int n, Complex<double> *in, double *out,
                                    unsigned int flags) {
  return fftw_plan_dft_c2r_1d(n, in, out, flags);
}
template <>
inline Plan<float> plan_dft_c2r_1d(int n, Complex<float> *in, float *out,
                                   unsigned int flags) {
  return fftwf_plan_dft_c2r_1d(n, in, out, flags);
}

} // namespace uspam::fftw

namespace uspam::fft {

// In memory cache with key type K and value type V
// additionally accepts a mutex to guard the V constructor
template <class Key, class Val> auto get_cached(Key key) {
  static thread_local std::unordered_map<Key, std::unique_ptr<Val>> _cache;

  auto &val = _cache[key];
  if (val == nullptr) {
    val = std::make_unique<Val>(key);
  }
  return val.get();
}

template <Floating T> struct engine_r2c_1d {
  std::span<T> real;
  std::span<fftw::Complex<T>> complex;
  fftw::Plan<T> plan;

  static auto get(size_t n) -> auto & {
    thread_local static auto cache = get_cached<size_t, engine_r2c_1d>;
    return *cache(n);
  }

  explicit engine_r2c_1d(size_t n)
      : real(fftw::alloc_real<T>(n), n),
        complex(fftw::alloc_complex<T>(n / 2 + 1), n / 2 + 1),
        plan(fftw::plan_dft_r2c_1d(static_cast<int>(n), real.data(),
                                   complex.data(), FFTW_ESTIMATE)) {}
  engine_r2c_1d(const engine_r2c_1d &) = delete;
  auto operator=(const engine_r2c_1d &) -> engine_r2c_1d & = delete;
  engine_r2c_1d(engine_r2c_1d &&) = delete;
  auto operator=(engine_r2c_1d &&) -> engine_r2c_1d & = delete;
  ~engine_r2c_1d() {
    fftw::destroy_plan<T>(plan);
    fftw::free<T>(real.data());
    fftw::free<T>(complex.data());
  }

  inline void execute() const { fftw::execute<T>(plan); }
};

template <Floating T> struct engine_c2r_1d {
  std::span<T> real;
  std::span<fftw::Complex<T>> complex;
  fftw::Plan<T> plan;

  static auto get(size_t n) -> auto & {
    thread_local static auto cache = get_cached<size_t, engine_c2r_1d>;
    return *cache(n);
  }

  explicit engine_c2r_1d(size_t n)
      : real(fftw::alloc_real<T>(n), n),
        complex(fftw::alloc_complex<T>(n / 2 + 1), n / 2 + 1),
        plan(fftw::plan_dft_c2r_1d<T>(static_cast<int>(n), complex.data(),
                                      real.data(), FFTW_ESTIMATE)) {}
  engine_c2r_1d(const engine_c2r_1d &) = delete;
  auto operator=(const engine_c2r_1d &) -> engine_c2r_1d & = delete;
  engine_c2r_1d(engine_c2r_1d &&) = delete;
  auto operator=(engine_c2r_1d &&) -> engine_c2r_1d & = delete;
  ~engine_c2r_1d() {
    fftw::destroy_plan<T>(plan);
    fftw::free<T>(real.data());
    fftw::free<T>(complex.data());
  }

  inline void execute() const { fftw::execute<T>(plan); }
};

template <typename T> struct engine_half_cx_1d {
  std::span<T> real;
  std::span<fftw::Complex<T>> complex;
  fftw::Plan<T> plan_f;
  fftw::Plan<T> plan_b;

  static auto get(size_t n) -> auto & {
    const auto cache = get_cached<size_t, engine_half_cx_1d>;
    return *cache(n);
  }

  explicit engine_half_cx_1d(size_t n)
      : real(fftw::alloc_real<T>(n), n),
        complex(fftw::alloc_complex<T>(n / 2 + 1), n / 2 + 1),
        plan_f(fftw::plan_dft_r2c_1d(static_cast<int>(n), real.data(),
                                     complex.data(), FFTW_ESTIMATE)),
        plan_b(fftw::plan_dft_c2r_1d(static_cast<int>(n), complex.data(),
                                     real.data(), FFTW_ESTIMATE)) {}
  engine_half_cx_1d(const engine_half_cx_1d &) = default;
  auto operator=(const engine_half_cx_1d &) -> engine_half_cx_1d & = default;
  engine_half_cx_1d(engine_half_cx_1d &&) = delete;
  auto operator=(engine_half_cx_1d &&) -> engine_half_cx_1d & = delete;
  ~engine_half_cx_1d() {
    fftw::destroy_plan<T>(plan_f);
    fftw::destroy_plan<T>(plan_b);
    fftw::free<T>(real.data());
    fftw::free<T>(complex.data());
  }

  inline void execute_r2c() const { fftw::execute<T>(plan_f); }
  inline void execute_c2r() const { fftw::execute<T>(plan_b); }
};

template <typename T> struct engine_1d {
  std::span<fftw::Complex<T>> in;
  std::span<fftw::Complex<T>> out;
  fftw::Plan<T> plan_f;
  fftw::Plan<T> plan_b;

  static auto get(size_t n) -> auto & {
    thread_local static auto cache = get_cached<size_t, engine_1d>;
    return *cache(n);
  }

  explicit engine_1d(size_t n)
      : in(fftw::alloc_complex<T>(n), n), out(fftw::alloc_complex<T>(n), n),
        plan_f(fftw::plan_dft_1d<T>(static_cast<int>(n), in.data(), out.data(),
                                    FFTW_FORWARD, FFTW_ESTIMATE)),
        plan_b(fftw::plan_dft_1d<T>(static_cast<int>(n), out.data(), in.data(),
                                    FFTW_BACKWARD, FFTW_ESTIMATE)) {}
  engine_1d(const engine_1d &) = default;
  auto operator=(const engine_1d &) -> engine_1d & = default;
  engine_1d(engine_1d &&) = delete;
  auto operator=(engine_1d &&) -> engine_1d & = delete;
  ~engine_1d() {
    fftw::destroy_plan<T>(plan_f);
    fftw::destroy_plan<T>(plan_b);
    fftw::free<T>(in.data());
    fftw::free<T>(out.data());
  }

  inline void execute_forward() const { fftw::execute<T>(plan_f); }
  inline void execute_backward() const { fftw::execute<T>(plan_b); }
};

} // namespace uspam::fft
