#pragma once

#include <QWidget>
#include <qcustomplot.h>

class AlinePlot : public QWidget {
  Q_OBJECT
public:
  AlinePlot(QWidget *parent = nullptr);

private:
  QCustomPlot *customPlot;
};