#pragma once

#include <QVector>
#include <QWidget>
#include <qcustomplot.h>
#include <span>

class AlinePlot : public QWidget {
  Q_OBJECT
public:
  AlinePlot(QWidget *parent = nullptr);

  auto plot() { return customPlot; }

public slots:
  void plot(std::span<double> x, std::span<double> y);

private:
  QCustomPlot *customPlot;

  QVector<double> m_x;
  QVector<double> m_y;
};