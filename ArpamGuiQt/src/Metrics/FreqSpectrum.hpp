#pragma once

#include "uspam/fft.hpp"
#include <DataProcWorker.hpp>
#include <QVector>
#include <QWidget>
#include <algorithm>
#include <qcustomplot.h>
#include <span>

class FreqSpectrum : public QWidget {
  Q_OBJECT
public:
  static constexpr int TICK_LENGTH = 5;
  static constexpr int SUBTICK_LENGTH = 3;

  explicit FreqSpectrum(QWidget *parent = nullptr);

  // Plot the FFT of the data
  template <uspam::Floating T>
  void setData(std::span<const T> data, double Fs_MHz = 1.0);
  void setData(const QVector<double> &data, double Fs_MHz = 1.0);

private:
  QCustomPlot *customPlot;
};