#pragma once

#include <QVector>
#include <QWidget>
#include <qcustomplot.h>
#include <span>

class AlinePlot : public QWidget {
  Q_OBJECT
public:
  using FloatType = double;

  AlinePlot(QWidget *parent = nullptr);

  auto getPlot() { return customPlot; }

public slots:
  void plot(std::span<const FloatType> x, std::span<const FloatType> y);

private:
  QCustomPlot *customPlot;

  QVector<FloatType> m_x;
  QVector<FloatType> m_y;
};