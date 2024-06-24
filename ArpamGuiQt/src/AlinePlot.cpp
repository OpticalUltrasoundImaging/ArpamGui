#include "AlinePlot.hpp"
#include <QBrush>
#include <QColor>
#include <QPen>
#include <QVBoxLayout>
#include <QVector>
#include <qcustomplot.h>
#include <span>

AlinePlot::AlinePlot(QWidget *parent)
    : QWidget(parent), customPlot(new QCustomPlot) {

  // Set dark background color
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
  m_x = QVector<double>(101);
  m_y = QVector<double>(101);
  for (int i = 0; i < 101; ++i) {
    m_x[i] = i / 50.0 - 1;    // x goes from -1 to 1
    m_y[i] = m_x[i] * m_x[i]; // let's plot a quadratic function
  }

  // create graph and assign data to it:
  customPlot->addGraph();
  customPlot->graph(0)->setData(m_x, m_y, true);
  customPlot->graph(0)->setPen(QPen(Qt::green));

  // give the axes some labels:
  customPlot->xAxis->setLabel("Samples");
  customPlot->yAxis->setLabel("Signal");

  // set axes ranges, so we see all data:
  customPlot->xAxis->setRange(-1, 1);
  customPlot->yAxis->setRange(0, 1);
  customPlot->replot();

  // UI
  auto *layout = new QVBoxLayout;
  setLayout(layout);

  layout->addWidget(customPlot);
}

void AlinePlot::plot(std::span<double> x, std::span<double> y) {
  assert(x.size() == y.size());
  assert(m_x.size() == x.size());
  assert(m_y.size() == x.size());

  std::copy(x.begin(), x.end(), m_x.data());
  std::copy(y.begin(), y.end(), m_y.data());

  // TODO Need to schedule replot
  update();
}