#include "AScanPlot.hpp"
#include <QBrush>
#include <QColor>
#include <QPen>
#include <QVBoxLayout>
#include <QVector>
#include <cmath>
#include <numbers>
#include <qcustomplot.h>
#include <span>

AScanPlot::AScanPlot(QWidget *parent)
    : QWidget(parent), customPlot(new QCustomPlot) {

  /*
   * Setup the customPlot
   */
  {

    // Set dark background color
    customPlot->setBackground(QColor(0, 0, 0)); // Black

    // Customize plot appearance
    const auto styleAxis = [](QCPAxis *axis) {
      axis->setBasePen(QPen(Qt::white));
      axis->setTickPen(QPen(Qt::white));
      axis->setSubTickPen(QPen(Qt::white));
      axis->setTickLabelColor(Qt::white);
      axis->setLabelColor(Qt::white);
    };
    styleAxis(customPlot->xAxis);
    styleAxis(customPlot->xAxis2);
    styleAxis(customPlot->yAxis);
    styleAxis(customPlot->yAxis2);

    customPlot->legend->setBrush(QBrush(QColor(255, 255, 255, 100)));

    // generate some data
    {
      constexpr int N = 201;
      m_x = QVector<FloatType>(N);
      m_y = QVector<FloatType>(N);
      for (int i = 0; i < N; ++i) {
        constexpr FloatType fct = std::numbers::pi * 2 / N;
        m_x[i] = i * fct;          // x goes from 0 to 2pi
        m_y[i] = std::sin(m_x[i]); // plot sine wave
      }
    }

    // create graph and assign data to it:
    customPlot->addGraph();
    customPlot->graph(0)->setData(m_x, m_y, true);
    customPlot->graph(0)->setPen(QPen(Qt::green));

    // set axes ranges
    customPlot->xAxis->setRange(m_x.front(), m_x.back());
    customPlot->yAxis->setRange(-1, 1);

    // give the axes some labels:
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");

    // Remove grid
    customPlot->xAxis->grid()->setVisible(false);
    customPlot->yAxis->grid()->setVisible(false);

    // Add top and right axis to have top and left border
    customPlot->xAxis2->setVisible(true);
    customPlot->yAxis2->setVisible(true);
    customPlot->xAxis2->setTickLabels(false);
    customPlot->yAxis2->setTickLabels(false);

    // Set ticks to point outwards
    customPlot->xAxis->setTickLength(0, 5);
    customPlot->xAxis->setSubTickLength(0, 3);
    customPlot->yAxis->setTickLength(0, 5);
    customPlot->yAxis->setSubTickLength(0, 3);

    customPlot->xAxis2->setTickLength(0, 0);
    customPlot->xAxis2->setSubTickLength(0, 0);
    customPlot->yAxis2->setTickLength(0, 0);
    customPlot->yAxis2->setSubTickLength(0, 0);

    customPlot->replot();

    // Interaction
    // customPlot->setInteraction(QCP::iRangeZoom, true);
    customPlot->setInteraction(QCP::iSelectItems, true);
    customPlot->setInteraction(QCP::iSelectPlottables, true);
  }

  // UI
  {
    auto *layout = new QVBoxLayout;
    setLayout(layout);

    layout->addWidget(customPlot);

    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
  }
}

void AScanPlot::plot(std::span<const FloatType> x,
                     std::span<const FloatType> y) {
  assert(x.size() == y.size());

  if (x.size() != m_x.size()) {
    m_x.resize(x.size());
    m_y.resize(x.size());
  }

  std::copy(x.begin(), x.end(), m_x.data());
  std::copy(y.begin(), y.end(), m_y.data());

  // replot
  customPlot->graph(0)->setData(m_x, m_y, true);
  customPlot->xAxis->setRange(x.front(), x.back());
  customPlot->yAxis->setRange(-1, 1);
  customPlot->replot();
}

void AScanPlot::showPointToolTip(QMouseEvent *event) {
  const auto pos = event->pos();
  const auto x = customPlot->xAxis->pixelToCoord(pos.x());
  const auto y = customPlot->yAxis->pixelToCoord(pos.y());

  setToolTip(QString("%1, %2").arg(x).arg(y));
}