#pragma once

#include <DataProcWorker.hpp>
#include <QMouseEvent>
#include <QVector>
#include <QWidget>
#include <ReconParamsController.hpp>
#include <memory>
#include <qcustomplot.h>
#include <span>
#include <type_traits>

class AScanPlot : public QWidget {
  Q_OBJECT
public:
  using FloatType = double;

  explicit AScanPlot(ReconParamsController *reconParams,
                     QWidget *parent = nullptr);

  auto getPlot() { return customPlot; }

  void plot(std::span<const FloatType> x, std::span<const FloatType> y);

  template <typename T> void plot(std::span<const T> y) {

    ensureX(y.size());

    // Ensure m_y size
    if (y.size() != m_y.size()) {
      m_y.resize(y.size());
    }

    if constexpr (std::is_same_v<T, FloatType>) {
      std::copy(y.begin(), y.end(), m_y.data());
    } else {
      for (int i = 0; i < y.size(); ++i) {
        m_y[i] = static_cast<FloatType>(y[i]);
      }
    }

    // replot
    customPlot->graph(0)->setData(m_x, m_y, true);
    customPlot->xAxis->setRange(m_x.front(), m_x.back());
    customPlot->yAxis->setRange(-1, 1);
    customPlot->replot();
  }

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
};