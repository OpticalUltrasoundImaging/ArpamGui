#pragma once

#include "DataProcWorker.hpp"
#include "Metrics/FWHM.hpp"
#include <QMouseEvent>
#include <QString>
#include <QVector>
#include <QWidget>
#include <ReconParamsController.hpp>
#include <memory>
#include <qcustomplot.h>
#include <span>
#include <type_traits>

class AScanFWHMTracers {
public:
  explicit AScanFWHMTracers(QCustomPlot *customPlot)
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
                     int graphIdx = 0);

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

class AScanPlot : public QWidget {
  Q_OBJECT
public:
  using FloatType = double;

  static constexpr FloatType MM_PER_PIXEL_PA = (1500.0 * 1000) / (180e6);
  static constexpr FloatType MM_PER_PIXEL_US = (1500.0 / 2 * 1000) / (180e6);

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

  struct PlotMeta {
    bool autoScaleY{false};
    FloatType yMin{-1.0};
    FloatType yMax{1.0};

    FloatType xScaler{1.0};
    QString xUnit{}; // If xUnit is not empty, use xScaler
  };

  explicit AScanPlot(ReconParamsController *reconParams,
                     QWidget *parent = nullptr);

  auto getPlot() { return customPlot; }

  void plot(const QVector<FloatType> &x, const QVector<FloatType> &y,
            const PlotMeta &meta);

  template <typename T> void plot(std::span<const T> y, const PlotMeta &meta);

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
  QLabel *m_FWHMLabel;
};