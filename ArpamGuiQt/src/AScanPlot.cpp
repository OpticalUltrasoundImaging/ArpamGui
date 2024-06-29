#include "AScanPlot.hpp"
#include <QBrush>
#include <QButtonGroup>
#include <QColor>
#include <QGridLayout>
#include <QPen>
#include <QRadioButton>
#include <QSplitter>
#include <QVBoxLayout>
#include <QVector>
#include <cmath>
#include <numbers>
#include <qcustomplot.h>
#include <qnamespace.h>
#include <qsizepolicy.h>
#include <qwidget.h>
#include <span>
#include <uspam/reconParams.hpp>

AScanPlot::AScanPlot(ReconParamsController *reconParams, QWidget *parent)
    : QWidget(parent), m_reconParams(reconParams), customPlot(new QCustomPlot) {

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
     * Plot selector
     */
    {
      auto *layout = new QGridLayout(this);

      auto *w = new QWidget;
      w->setLayout(layout);
      splitter->addWidget(w);

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
      layout->addWidget(radioRFRaw, 0, 0);

      layout->addWidget(radioRFBeamformedPA, 1, 0);
      layout->addWidget(radioRFEnvPA, 2, 0);
      layout->addWidget(radioRFLogPA, 3, 0);

      layout->addWidget(radioRFBeamformedUS, 1, 1);
      layout->addWidget(radioRFEnvUS, 2, 1);
      layout->addWidget(radioRFLogUS, 3, 1);

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

    // Stretchable spacer
    splitter->addWidget(new QWidget);

    // layout->setSpacing(0);
    // layout->setContentsMargins(0, 0, 0, 0);
  }

  this->setSizePolicy(QSizePolicy::MinimumExpanding,
                      QSizePolicy::MinimumExpanding);
  this->setMinimumSize(QSize{500, 500});
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

void AScanPlot::ensureX(int size) {
  if (m_x.size() != size) {
    m_x.resize(size);
    std::iota(m_x.begin(), m_x.end(), 0);
  }
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

    const auto msg = QString("Select AScan: %1. Flip: %2")
                         .arg(idx)
                         .arg(flip ? "true" : "false");

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
    customPlot->yAxis->setLabel("Signal");
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, true);
  } break;
  case PlotType::RFEnvUS: {
    // US rfEnv
    const auto &rf = m_data->US.rfEnv;
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal");
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, true);
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