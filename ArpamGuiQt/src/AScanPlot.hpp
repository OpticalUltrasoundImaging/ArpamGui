#pragma once

#include <QVector>
#include <QWidget>
#include <qcustomplot.h>
#include <span>

class AScanPlot : public QWidget {
  Q_OBJECT
public:
  using FloatType = double;

  explicit AScanPlot(QWidget *parent = nullptr);

  auto getPlot() { return customPlot; }

public slots:
  void plot(std::span<const FloatType> x, std::span<const FloatType> y);

private:
  QCustomPlot *customPlot;

  QVector<FloatType> m_x;
  QVector<FloatType> m_y;
};