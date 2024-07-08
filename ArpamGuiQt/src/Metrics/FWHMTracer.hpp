#pragma once

#include "Metrics/FWHM.hpp"
#include <qcustomplot.h>

class FWHMTracers {
public:
  explicit FWHMTracers(QCustomPlot *customPlot)
      : customPlot(customPlot), peakTracer(new QCPItemTracer(customPlot)),
        lineLower(new QCPItemLine(customPlot)),
        lineUpper(new QCPItemLine(customPlot)) {
    peakTracer->setInterpolating(true);
    peakTracer->setStyle(QCPItemTracer::tsCircle);
    peakTracer->setPen(QPen(Qt::red));
    peakTracer->setBrush(Qt::red);

    const auto LineColor = Qt::blue;
    const auto pen = QPen(LineColor);
    lineLower->setPen(pen);
    lineUpper->setPen(pen);
  }

  template <typename T>
  FWHM<T> updateData(const QVector<T> &x, const QVector<T> &y,
                     int graphIdx = 0) {
    const auto fwhm = calcFWHM<T>(x, y);

    peakTracer->setGraph(customPlot->graph(graphIdx));
    peakTracer->setGraphKey(x[fwhm.peakIdx]);

    lineLower->start->setCoords(fwhm.lowerX, 0);
    lineLower->end->setCoords(fwhm.lowerX, y[fwhm.lowerIdx]);

    lineUpper->start->setCoords(fwhm.upperX, 0);
    lineUpper->end->setCoords(fwhm.upperX, y[fwhm.upperIdx]);
    return fwhm;
  }

  void toggle() {
    peakTracer->setVisible(!peakTracer->visible());
    lineLower->setVisible(!lineLower->visible());
    lineUpper->setVisible(!lineUpper->visible());
  }

private:
  QCustomPlot *customPlot;
  QCPItemTracer *peakTracer;
  QCPItemLine *lineLower;
  QCPItemLine *lineUpper;
};