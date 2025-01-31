#pragma once

#include <algorithm>
#include <armadillo>
#include <array>
#include <concepts>
#include <fftconv/fftconv.hpp>
#include <kfr/dsp.hpp>
#include <kfr/dsp/iir_design.hpp>
#include <kfr/kfr.h>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <span>
#include <uspam/signal.hpp>


namespace uspam {

template <typename T> class Filter {
public:
  using value_type = T;

  virtual ~Filter() = default;
  Filter() = default;
  Filter(const Filter &) = default;
  Filter(Filter &&) = default;
  Filter &operator=(const Filter &) = default;
  Filter &operator=(Filter &&) = default;

  virtual void forward(std::span<const T> input, std::span<T> output) const = 0;
};

template <typename T>
concept IsFilterSubclass = std::derived_from<T, Filter<typename T::value_type>>;

template <typename T> class FIRFilter final : public Filter<T> {
public:
  FIRFilter(int numtaps, const T bandpassLowFreq, const T bandpassHighFreq)
      : FIRFilter(numtaps,
                  std::array<double, 6>{0, bandpassLowFreq, bandpassLowFreq,
                                        bandpassHighFreq, bandpassHighFreq, 1},
                  std::array<double, 6>{0, 0, 1, 1, 0, 0}) {}

  FIRFilter(int numtaps, const std::span<const double> freq,
            const std::span<const double> gain) {
    const auto _kernel = uspam::signal::firwin2<double>(numtaps, freq, gain);
    kernel = arma::conv_to<arma::Col<T>>::from(_kernel);
  }

  explicit FIRFilter(arma::Col<T> kernel) : kernel(std::move(kernel)) {}

  void forward(std::span<const T> input, std::span<T> output) const override {
    fftconv::oaconvolve_fftw<T, fftconv::Same>(input, kernel, output);

    // Clear invalid regions
    const auto invalidSize = kernel.size() / 2;
    std::ranges::fill(output.subspan(0, invalidSize), 0);
    std::ranges::fill(output.subspan(output.size() - invalidSize, invalidSize),
                      0);
  }

private:
  arma::Col<T> kernel;
};

template <typename T> class ButterworthFilter final : public Filter<T> {
public:
  ButterworthFilter(int order, const T bandpassLowFreq,
                    const T bandpassHighFreq)
      : bqs(kfr::to_sos(iir_bandpass(kfr::butterworth<T>(order),
                                     bandpassLowFreq, bandpassHighFreq))) {}

  void forward(std::span<const T> input, std::span<T> output) const override {
    const auto input_ = kfr::make_univector(input.data(), input.size());
    auto output_ = kfr::make_univector(output.data(), output.size());
    output_ = kfr::iir(input_, bqs);
  }

private:
  kfr::iir_params<T> bqs;
};

template <typename T>
void applyMedianFilter(arma::Mat<T> &mat, int medFiltSize) {
  const int ksize = (medFiltSize % 2 == 0) ? medFiltSize + 1 : medFiltSize;
  cv::Mat cvMat(mat.n_cols, mat.n_rows, cv::traits::Type<T>::value,
                (void *)mat.memptr());
  cv::medianBlur(cvMat, cvMat, ksize);
  mat = arma::Mat<T>(cvMat.ptr<T>(), mat.n_rows, mat.n_cols, true);
}
} // namespace uspam
