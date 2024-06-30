#include "AScanPlot.hpp"
#include <QBrush>
#include <QButtonGroup>
#include <QColor>
#include <QFrame>
#include <QGridLayout>
#include <QPen>
#include <QRadioButton>
#include <QSplitter>
#include <QVBoxLayout>
#include <QVector>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <qboxlayout.h>
#include <qcustomplot.h>
#include <qnamespace.h>
#include <qpushbutton.h>
#include <qsizepolicy.h>
#include <qwidget.h>
#include <span>
#include <uspam/reconParams.hpp>

AScanPlot::AScanPlot(ReconParamsController *reconParams, QWidget *parent)
    : QWidget(parent), m_reconParams(reconParams), customPlot(new QCustomPlot),
      m_FWHMtracers(customPlot) {

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

    // create graph and assign data to it:
    customPlot->addGraph();
    customPlot->graph(0)->setPen(QPen(Qt::green));

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

    // generate some data
    {
      constexpr int N = 201;

      QVector<FloatType> x(N);
      QVector<FloatType> y(N);
      for (int i = 0; i < N; ++i) {
        constexpr FloatType fct = std::numbers::pi * 2 / N;
        x[i] = i * fct;        // x goes from 0 to 2pi
        y[i] = std::sin(x[i]); // plot sine wave
      }
      plot(x, y);
    }

    // Interaction
    // customPlot->setInteraction(QCP::iRangeZoom, true);
    customPlot->setInteraction(QCP::iSelectItems, true);
    customPlot->setInteraction(QCP::iSelectPlottables, true);

    customPlot->setMinimumHeight(200);
  }

  // UI
  {
    auto *layout = new QVBoxLayout;
    setLayout(layout);

    auto *splitter = new QSplitter;
    splitter->setOrientation(Qt::Vertical);
    layout->addWidget(splitter);

    /*
     * Plot
     */
    customPlot->setSizePolicy(QSizePolicy::MinimumExpanding,
                              QSizePolicy::MinimumExpanding);
    splitter->addWidget(customPlot);
    splitter->setStretchFactor(0, 1);

    /*
     * Plot Controls
     */
    {
      auto *w = new QWidget;
      splitter->addWidget(w);

      auto *layout = new QVBoxLayout;
      w->setLayout(layout);
      w->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Maximum);

      // Plot controls
      {
        auto *btn = new QPushButton("Toggle FWHM");
        btn->setCheckable(true);
        btn->setChecked(false);
        connect(btn, &QPushButton::clicked, [this](bool checked) {
          m_FWHMtracers.toggle();
          customPlot->replot();
        });
        layout->addWidget(btn);
      }

      // Horizontal line separator
      {
        auto *line = new QFrame;
        line->setFrameShape(QFrame::Shape::HLine);
        line->setFrameShadow(QFrame::Sunken);
        layout->addWidget(line);
      }

      // Plot Type Selectors
      {
        auto *grid = new QGridLayout(this);
        layout->addLayout(grid);

        // Create a button group
        auto *group = new QButtonGroup(this);
        group->setExclusive(true);

        // Create radio buttons for each enum value
        auto *radioRFRaw = new QRadioButton("RF Raw (PA + US)");
        auto *radioRFBeamformedUS = new QRadioButton("RF Beamformed US");
        auto *radioRFBeamformedPA = new QRadioButton("RF Beamformed PA");
        auto *radioRFEnvUS = new QRadioButton("RF Env US");
        auto *radioRFEnvPA = new QRadioButton("RF Env PA");
        auto *radioRFLogUS = new QRadioButton("RF Log US");
        auto *radioRFLogPA = new QRadioButton("RF Log PA");

        // Add buttons to the layout and button group
        grid->addWidget(radioRFRaw, 0, 0);

        grid->addWidget(radioRFBeamformedPA, 1, 0);
        grid->addWidget(radioRFEnvPA, 2, 0);
        grid->addWidget(radioRFLogPA, 3, 0);

        grid->addWidget(radioRFBeamformedUS, 1, 1);
        grid->addWidget(radioRFEnvUS, 2, 1);
        grid->addWidget(radioRFLogUS, 3, 1);

        group->addButton(radioRFRaw, static_cast<int>(PlotType::RFRaw));
        group->addButton(radioRFEnvPA, static_cast<int>(PlotType::RFEnvPA));
        group->addButton(radioRFEnvUS, static_cast<int>(PlotType::RFEnvUS));
        group->addButton(radioRFBeamformedPA,
                         static_cast<int>(PlotType::RFBeamformedPA));
        group->addButton(radioRFBeamformedUS,
                         static_cast<int>(PlotType::RFBeamformedUS));
        group->addButton(radioRFLogPA, static_cast<int>(PlotType::RFLogPA));
        group->addButton(radioRFLogUS, static_cast<int>(PlotType::RFLogUS));

        // Connect the button group's signal to a lambda function
        connect(group, &QButtonGroup::idClicked, [this](int id) {
          m_type = static_cast<PlotType>(id);
          plotCurrentAScan();
        });

        radioRFRaw->click();
      }
    }

    // Stretchable spacer
    splitter->addWidget(new QWidget);

    // layout->setSpacing(0);
    // layout->setContentsMargins(0, 0, 0, 0);
  }

  this->setSizePolicy(QSizePolicy::MinimumExpanding,
                      QSizePolicy::MinimumExpanding);
  this->setMinimumSize(QSize{500, 500});
}

void AScanPlot::showPointToolTip(QMouseEvent *event) {
  const auto pos = event->pos();
  const auto x = customPlot->xAxis->pixelToCoord(pos.x());
  const auto y = customPlot->yAxis->pixelToCoord(pos.y());

  setToolTip(QString("%1, %2").arg(x).arg(y));
}

void AScanPlot::ensureX(int size) {
  if (m_x.size() != size) {
    m_x.resize(size);
    std::iota(m_x.begin(), m_x.end(), 0);
  }
}

template <typename T>
void AScanPlot::plot(std::span<const T> y, bool autoScaleY, FloatType yMin,
                     FloatType yMax) {
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
  plot(m_x, m_y, autoScaleY, yMin, yMax);
}

void AScanPlot::plot(const QVector<FloatType> &x, const QVector<FloatType> &y,
                     bool autoScaleY, FloatType yMin, FloatType yMax) {
  assert(x.size() == y.size());

  // Compute FWHM
  m_FWHMtracers.updateData(x, y);

  // replot
  customPlot->graph(0)->setData(x, y, true);
  customPlot->xAxis->setRange(x.front(), x.back());
  if (autoScaleY) {
    const auto [min, max] = std::minmax_element(m_y.cbegin(), m_y.cend());
    yMin = *min;
    yMax = *max;
  }

  customPlot->yAxis->setRange(yMin, yMax);
  customPlot->replot();
}

void AScanPlot::plotCurrentAScan() {
  if (m_data == nullptr) [[unlikely]] {
    return;
  }

  // Correct for flip and rotation in the selected AScan idx
  // and store result in m_AScanPlotIdx
  {
    const bool flip = uspam::recon::ReconParams::flip(m_data->frameIdx);
    auto idx = m_AScanPlotIdx_canvas;
    if (flip) {
      idx -= m_reconParams->params.PA.rotateOffset;

      constexpr int AScansPerBScan = 1000;
      if (idx < 0) {
        idx += AScansPerBScan;
      } else if (idx >= AScansPerBScan) {
        idx -= AScansPerBScan;
      }

      idx = AScansPerBScan - 1 - idx;
    }

    const auto msg =
        QString("AScan: %1. Flip: %2").arg(idx).arg(flip ? "true" : "false");

    m_AScanPlotIdx = idx;
  }

  /*
   * Plot AScan
   */

  switch (m_type) {
  case PlotType::RFRaw: {
    // Original RF
    const auto &rf = m_data->rf;
    const std::span y{rf.colptr(m_AScanPlotIdx), rf.n_rows};
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    plot(y);
  } break;

  case PlotType::RFBeamformedPA: {
    // PA rfEnv
    const auto &rf = m_data->PA.rfBeamformed;
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;

  case PlotType::RFBeamformedUS: {
    // US rfEnv
    const auto &rf = m_data->US.rfBeamformed;
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;

  case PlotType::RFEnvPA: {
    // PA rfEnv
    const auto &rf = m_data->PA.rfEnv;
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;
  case PlotType::RFEnvUS: {
    // US rfEnv
    const auto &rf = m_data->US.rfEnv;
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;

  case PlotType::RFLogPA: {
    // US rfLog
    const auto &rf = m_data->PA.rfLog;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal");
    plot(y, false, 0, 256);
  } break;

  case PlotType::RFLogUS: {
    // US rfLog
    const auto &rf = m_data->US.rfLog;
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal");
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, false, 0, 256);
  } break;

  case Size:
    // NOOP
    break;
  }
}

void AScanPlot::handleAScanSelected(int idx) {
  // The index received here is in canvas pixmap coordinates (i.e. doesn't
  // account for flip and rotation offset)
  m_AScanPlotIdx_canvas = idx;

  plotCurrentAScan();
}