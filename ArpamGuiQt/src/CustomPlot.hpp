#pragma once

#include "Metrics/FWHMTracer.hpp"
#include <QWidget>
#include <qcustomplot.h>

/*
 * A custom, QCustomPlot with some QoL features
 */
class CustomPlot : public QCustomPlot {
  Q_OBJECT
public:
  static constexpr int TICK_LENGTH = 5;
  static constexpr int SUBTICK_LENGTH = 3;

  struct PlotMeta {
    QString name;

    bool autoScaleY{false};
    double yMin{-1.0};
    double yMax{1.0};

    double xMin{0};
    double xMax{0};

    double xScaler{1.0};
    QString xUnit; // If xUnit is not empty, use xScaler
  };

  explicit CustomPlot(QWidget *parent = nullptr);

  void plot(const QVector<double> &x, const QVector<double> &y,
            const CustomPlot::PlotMeta &meta);

  template <typename T>
  void plot(std::span<const T> y, const CustomPlot::PlotMeta &meta) {
    ensureX(y.size());

    // Ensure m_y size
    if (y.size() != m_y.size()) {
      m_y.resize(y.size());
    }

    if constexpr (std::is_same_v<T, double>) {
      std::copy(y.begin(), y.end(), m_y.data());
    } else {
      for (int i = 0; i < y.size(); ++i) {
        m_y[i] = static_cast<double>(y[i]);
      }
    }

    // replot
    plot(m_x, m_y, meta);
  }

  [[nodiscard]] auto FWHMLabel() const { return m_FWHMLabel; }

  [[nodiscard]] auto actResetZoom() const { return m_actResetZoom; }
  [[nodiscard]] auto actShowFWHM() const { return m_actShowFWHM; }
  [[nodiscard]] auto actShowLegend() const { return m_actShowLegend; }

public slots:
  void resetZoom();

private:
  // Makes sure m_x is properly generated
  void ensureX(int size);

  // FWHM markers
  FWHMTracers *m_FWHMtracers;
  QLabel *m_FWHMLabel;

  QAction *m_actResetZoom;
  QAction *m_actShowFWHM;
  QAction *m_actShowLegend;

  PlotMeta m_meta;
  QVector<double> m_x;
  QVector<double> m_y;
};