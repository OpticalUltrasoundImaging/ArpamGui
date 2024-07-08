#pragma once

#include "DataProcWorker.hpp"
#include "Metrics/FWHM.hpp"
#include "Metrics/FWHMTracer.hpp"
#include <QMouseEvent>
#include <QString>
#include <QVector>
#include <QWidget>
#include <ReconParamsController.hpp>
#include <memory>
#include <qcustomplot.h>
#include <span>

class AScanPlot : public QWidget {
  Q_OBJECT
public:
  using FloatType = double;
  static constexpr int TICK_LENGTH = 5;
  static constexpr int SUBTICK_LENGTH = 3;

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
    QString name;

    bool autoScaleY{false};
    FloatType yMin{-1.0};
    FloatType yMax{1.0};

    FloatType xMin{0};
    FloatType xMax{0};

    FloatType xScaler{1.0};
    QString xUnit{}; // If xUnit is not empty, use xScaler
  };

  explicit AScanPlot(ReconParamsController *reconParams,
                     QWidget *parent = nullptr);

  auto getPlot() { return customPlot; }

  void plot(const QVector<FloatType> &x, const QVector<FloatType> &y);

  template <typename T> void plot(std::span<const T> y);

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
  std::shared_ptr<BScanData<DataProcWorker::FloatType>> m_data;

  QCustomPlot *customPlot;
  PlotMeta m_plotMeta;
  QVector<FloatType> m_x;
  QVector<FloatType> m_y;

  int m_AScanPlotIdx_canvas{}; // Received from canvas, not corrected for flip
                               // and rotation
  int m_AScanPlotIdx{};        // Corrected for flip and rotation

  PlotType m_type{PlotType::RFRaw};

  // FWHM markers
  FWHMTracers m_FWHMtracers;
  QLabel *m_FWHMLabel;
};
