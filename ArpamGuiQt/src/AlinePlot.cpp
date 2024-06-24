#include "AlinePlot.hpp"
#include <QBrush>
#include <QColor>
#include <QPen>
#include <QVBoxLayout>
#include <QVector>
#include <qcustomplot.h>

AlinePlot::AlinePlot(QWidget *parent)
    : QWidget(parent), customPlot(new QCustomPlot) {

  // Set dark background color
  //   customPlot->setBackground(QColor(30, 30, 30)); // Dark gray
  customPlot->setBackground(QColor(0, 0, 0)); // Black

  // Customize plot appearance
  customPlot->xAxis->setBasePen(QPen(Qt::white));
  customPlot->yAxis->setBasePen(QPen(Qt::white));
  customPlot->xAxis->setTickPen(QPen(Qt::white));
  customPlot->yAxis->setTickPen(QPen(Qt::white));
  customPlot->xAxis->setSubTickPen(QPen(Qt::white));
  customPlot->yAxis->setSubTickPen(QPen(Qt::white));
  customPlot->xAxis->setTickLabelColor(Qt::white);
  customPlot->yAxis->setTickLabelColor(Qt::white);
  customPlot->xAxis->setLabelColor(Qt::white);
  customPlot->yAxis->setLabelColor(Qt::white);
  customPlot->legend->setBrush(QBrush(QColor(255, 255, 255, 100)));

  // generate some data:
  QVector<double> x(101), y(101); // initialize with entries 0..100
  for (int i = 0; i < 101; ++i) {
    x[i] = i / 50.0 - 1; // x goes from -1 to 1
    y[i] = x[i] * x[i];  // let's plot a quadratic function
  }

  // create graph and assign data to it:
  customPlot->addGraph();
  customPlot->graph(0)->setData(x, y);
  customPlot->graph(0)->setPen(QPen(Qt::green));

  // give the axes some labels:
  customPlot->xAxis->setLabel("x");
  customPlot->yAxis->setLabel("y");

  // set axes ranges, so we see all data:
  customPlot->xAxis->setRange(-1, 1);
  customPlot->yAxis->setRange(0, 1);
  customPlot->replot();

  // UI
  auto *layout = new QVBoxLayout;
  setLayout(layout);

  layout->addWidget(customPlot);
}