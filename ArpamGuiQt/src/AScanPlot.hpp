#pragma once

#include <DataProcWorker.hpp>
#include <QMouseEvent>
#include <QString>
#include <QVector>
#include <QWidget>
#include <ReconParamsController.hpp>
#include <memory>
#include <qcustomplot.h>
#include <span>
#include <type_traits>

struct FWHM {
  int peakIdx;
  int lowerIdx;
  int upperIdx;
};

template <typename T> auto calcFWHM(const QVector<T> &x, const QVector<T> &y) {
  double maxY = *std::max_element(y.constBegin(), y.constEnd());
  int peakIndex =
      std::distance(y.constBegin(), std::find(y.begin(), y.end(), maxY));
  const double halfMax = maxY / 2.0;

  int lowerIndex = peakIndex;
  int upperIndex = peakIndex;

  // Search for half max crossing points
  while (lowerIndex > 0 && y[lowerIndex] > halfMax) {
    lowerIndex--;
  }

  while (upperIndex < y.size() && y[upperIndex] > halfMax) {
    upperIndex++;
  }

  return FWHM{peakIndex, lowerIndex, upperIndex};
}

class AScanFWHMTracers {
public:
  explicit AScanFWHMTracers(QCustomPlot *customPlot)
      : customPlot(customPlot), peakTracer(new QCPItemTracer(customPlot)),
        lineLower(new QCPItemLine(customPlot)),
        lineUpper(new QCPItemLine(customPlot))

  {
    peakTracer->setInterpolating(true);
    peakTracer->setStyle(QCPItemTracer::tsCircle);
    peakTracer->setPen(QPen(Qt::red));
    peakTracer->setBrush(Qt::red);

    lineLower->setPen(QPen(Qt::blue));
    lineUpper->setPen(QPen(Qt::blue));
  }

  template <typename T>
  void updateData(const QVector<T> &x, const QVector<T> &y, int graphIdx = 0) {
    const auto fwhm = calcFWHM<T>(x, y);
    const auto lowerX = x[fwhm.lowerIdx];
    const auto upperX = x[fwhm.upperIdx];

    const auto halfMax = y[fwhm.peakIdx] / 2;

    peakTracer->setGraph(customPlot->graph(graphIdx));
    peakTracer->setGraphKey(x[fwhm.peakIdx]);

    lineLower->start->setCoords(lowerX, 0);
    lineLower->end->setCoords(lowerX, halfMax);

    lineUpper->start->setCoords(upperX, 0);
    lineUpper->end->setCoords(upperX, halfMax);
  }

  void toggle() {
    lineLower->setVisible(!lineLower->visible());
    lineUpper->setVisible(!lineUpper->visible());
    peakTracer->setVisible(!peakTracer->visible());
  }

private:
  QCustomPlot *customPlot;
  QCPItemTracer *peakTracer;
  QCPItemLine *lineLower;
  QCPItemLine *lineUpper;
};

class AScanPlot : public QWidget {
  Q_OBJECT
public:
  using FloatType = double;

  enum PlotType {
    RFRaw,
    RFBeamformedUS,
    RFBeamformedPA,
    RFEnvUS,
    RFEnvPA,
    RFLogUS,
    RFLogPA,
    Size
  };
  inline static const std::array<QString, Size> PlotTypeStr{
      "RF Raw", "RF Env US", "RF Env PA", "RF Log US", "RF Log PA"};

  explicit AScanPlot(ReconParamsController *reconParams,
                     QWidget *parent = nullptr);

  auto getPlot() { return customPlot; }

  void plot(const QVector<FloatType> &x, const QVector<FloatType> &y,
            bool autoScaleY = false, FloatType yMin = -1.0,
            FloatType yMax = 1.0);

  template <typename T>
  void plot(std::span<const T> y, bool autoScaleY = false,
            FloatType yMin = -1.0, FloatType yMax = 1.0);

public slots:

  void showPointToolTip(QMouseEvent *event);

  void setData(std::shared_ptr<BScanData<DataProcWorker::FloatType>> data) {
    m_data = std::move(data);
  }

  void plotCurrentAScan();

  void handleAScanSelected(int idx);

private:
  // Makes sure m_x is properly generated
  void ensureX(int size);

  ReconParamsController *m_reconParams;

  QCustomPlot *customPlot;

  QVector<FloatType> m_x;
  QVector<FloatType> m_y;

  std::shared_ptr<BScanData<DataProcWorker::FloatType>> m_data;

  int m_AScanPlotIdx_canvas{}; // Received from canvas, not corrected for flip
                               // and rotation
  int m_AScanPlotIdx{};        // Corrected for flip and rotation

  PlotType m_type{PlotType::RFRaw};

  // FWHM markers
  AScanFWHMTracers m_FWHMtracers;
};