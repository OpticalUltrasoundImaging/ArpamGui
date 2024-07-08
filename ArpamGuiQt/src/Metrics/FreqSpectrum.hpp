#pragma once

#include "uspam/fft.hpp"
#include <CustomPlot.hpp>
#include <DataProcWorker.hpp>
#include <QVector>
#include <QWidget>
#include <algorithm>
#include <qcustomplot.h>
#include <span>

class FreqSpectrum : public QWidget {
  Q_OBJECT
public:
  explicit FreqSpectrum(QWidget *parent = nullptr);

  // Plot the FFT of the data
  template <uspam::Floating T>
  void setData(std::span<const T> data, double Fs_MHz = 1.0);
  void setData(const QVector<double> &data, double Fs_MHz = 1.0);

private:
  CustomPlot *customPlot;
};