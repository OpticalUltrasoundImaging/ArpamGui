#include "Metrics/FreqSpectrum.hpp"
#include "PlotCommon.hpp"
#include <QPen>
#include <QVBoxLayout>
#include <QVector>
#include <Qt>
#include <algorithm>
#include <qcustomplot.h>
#include <tuple>
#include <type_traits>

FreqSpectrum::FreqSpectrum(QWidget *parent) : customPlot(new CustomPlot) {

  /*
   * Setup the AScan customPlot
   */
  {
    customPlot->xAxis->setLabel("Frequency (MHz)");
    customPlot->yAxis->setLabel("Magnitude");
  }

  /*
   * UI
   */
  {
    auto *layout = new QVBoxLayout;
    setLayout(layout);

    layout->addWidget(customPlot);
  }
}

void FreqSpectrum::setData(const QVector<double> &data, const double Fs_MHz) {
  std::span sp(data.begin(), data.size());
  setData(sp, Fs_MHz);
}

template <uspam::Floating Tin, uspam::Floating Tout>
auto dbfft(const std::span<const Tin> y, const Tin fs) {
  const auto N = y.size();

  // Calculate real FFT and frequency vector
  const auto Nsp = N / 2 + 1;

  // freq = np.arange((N / 2) + 1) / (float(N) / fs);
  QVector<Tout> freq;
  freq.resize(Nsp);
  const auto fct = static_cast<Tout>(fs / N);
  for (int i = 0; i < Nsp; ++i) {
    freq[i] = i * fct;
  }

  // Scale the magnitude of FFT by window and factor
  // of 2, because we are using half of FFT spectrum.
  // sp = np.fft.rfft(x)
  // s_mag = np.abs(sp) * 2 / np.sum(win);
  uspam::fft::engine_r2c_1d<Tin> engine(N);
  std::copy(y.begin(), y.end(), engine.real.begin());
  engine.execute();
  QVector<Tout> sp;
  sp.resize(Nsp);
  for (int i = 0; i < Nsp; ++i) {
    const auto &cx = engine.complex[i];
    const std::complex<Tin> _cx(cx[0], cx[1]);

    const auto sp_mag = std::abs(_cx) / N;

    // Convert to dBFS
    // s_dbfs = 20 * np.log10(s_mag / ref);
    // const auto sp_dbfs = static_cast<Tout>(20 * std::log10(sp_mag));

    sp[i] = sp_mag;
  }

  return std::tuple{freq, sp};
}

template <uspam::Floating T>
void FreqSpectrum::setData(std::span<const T> data, const double Fs_MHz) {

  const auto [freq, sp] = dbfft<T, double>(data, Fs_MHz);

  customPlot->graph(0)->setData(freq, sp, true);

  customPlot->xAxis->setRange(0, freq.back());
  customPlot->xAxis->ticker()->setTickCount(10);

  const auto [min, max] = std::minmax_element(sp.cbegin(), sp.cend());
  customPlot->yAxis->setRange(0, *max);

  customPlot->replot();
}

// Explicit instantiation of the member function template above
template void FreqSpectrum::setData(std::span<const float>, double);
template void FreqSpectrum::setData(std::span<const double>, double);
