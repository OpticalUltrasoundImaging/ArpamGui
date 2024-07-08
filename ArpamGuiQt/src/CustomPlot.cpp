#include "CustomPlot.hpp"
#include "PlotCommon.hpp"
#include <qcustomplot.h>

CustomPlot::CustomPlot(QWidget *parent) : QCustomPlot(parent) {
  auto *customPlot = this;

  // Appearance
  {
    // Set dark background color
    customPlot->setBackground(QColor(0, 0, 0)); // Black

    // Style axis
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

    customPlot->xAxis2->setVisible(true);
    customPlot->yAxis2->setVisible(true);

    // customPlot->xAxis->grid()->setVisible(false);
    // customPlot->yAxis->grid()->setVisible(false);

    // NOLINTBEGIN(*-magic-numbers)
    customPlot->legend->setBrush(QBrush(QColor(255, 255, 255, 100)));
    customPlot->legend->setVisible(true);

    setupAxis(customPlot->xAxis, {}, true, 0, TICK_LENGTH, 0, SUBTICK_LENGTH);
    setupAxis(customPlot->yAxis, {}, true, 0, TICK_LENGTH, 0, SUBTICK_LENGTH);
    setupAxis(customPlot->xAxis2, {}, false);
    setupAxis(customPlot->yAxis2, {}, false);

    customPlot->setMinimumHeight(200);
    // NOLINTEND(*-magic-numbers)
  }

  // Interactions
  {
    // customPlot->setMouseTracking(true);

    // This line fails on MSVC for some reason
    // connect(customPlot, &QCustomPlot::mouseMove, this,
    //         [this](QMouseEvent *event) {
    //           const auto x =
    //           customPlot->xAxis->pixelToCoord(event->pos().x()); const
    //           auto y = customPlot->yAxis->pixelToCoord(event->pos().y());

    //           const auto txt = QString("%1 , %2").arg(x).arg(y, 6, 'f',
    //           4); QToolTip::showText(QCursor::pos(), txt);
    //         });

    customPlot->setInteraction(QCP::iRangeZoom, true);
    customPlot->setInteraction(QCP::iRangeDrag, true);
    customPlot->setInteraction(QCP::iSelectItems, true);
    customPlot->setInteraction(QCP::iSelectPlottables, true);

    customPlot->setSizePolicy(QSizePolicy::MinimumExpanding,
                              QSizePolicy::MinimumExpanding);
  }

  // Add graph 0
  {
    customPlot->addGraph();
    customPlot->graph(0)->setPen(QPen(Qt::green));
  }
}
