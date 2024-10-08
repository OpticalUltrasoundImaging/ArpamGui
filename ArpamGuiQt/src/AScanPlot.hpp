#pragma once

#include "Common.hpp"
#include "CustomPlot.hpp"
#include "Metrics/FreqSpectrum.hpp"
#include "RFBuffer.hpp"
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

  template <typename T>
  void plot(std::span<const T> y, const CustomPlot::PlotMeta &meta) {
    constexpr double Fs_MHz = 180;

    if constexpr (std::is_same_v<T, double>) {
      customPlot->plot(y, meta);

      // Update freq spectrum from ALine
      m_freqSpectrum->setData<T>(y, Fs_MHz);
    } else {
      QVector<double> _y;
      _y.resize(y.size());
      std::copy(y.begin(), y.end(), _y.begin());

      std::span<const double> yspan{_y.begin(), static_cast<size_t>(_y.size())};

      customPlot->plot(yspan, meta);

      // Update freq spectrum from ALine
      m_freqSpectrum->setData(yspan, Fs_MHz);
    }
  }

  inline void plot(const QVector<double> &x, const QVector<double> &y,
                   const CustomPlot::PlotMeta &meta) {
    customPlot->plot(x, y, meta);

    // Update freq spectrum from ALine
    constexpr double Fs_MHz = 180;
    m_freqSpectrum->setData(y, Fs_MHz);
  }

public slots:

  void showPointToolTip(QMouseEvent *event);

  void setData(std::shared_ptr<BScanData<ArpamFloat>> data) {
    m_data = std::move(data);
  }

  void plotCurrentAScan();

  void handleAScanSelected(int idx);

private:
  ReconParamsController *m_reconParams;
  std::shared_ptr<BScanData<ArpamFloat>> m_data;

  CustomPlot *customPlot;
  QVector<double> m_x;
  QVector<double> m_y;

  int m_AScanPlotIdx_canvas{}; // Received from canvas, not corrected for flip
                               // and rotation
  int m_AScanPlotIdx{};        // Corrected for flip and rotation

  PlotType m_type{PlotType::RFRaw};
  PlotType m_lastType{PlotType::Size};

  // Freq plot
  FreqSpectrum *m_freqSpectrum;
};
