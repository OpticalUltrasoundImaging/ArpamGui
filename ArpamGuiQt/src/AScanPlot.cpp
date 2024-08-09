#include "AScanPlot.hpp"
#include "CustomPlot.hpp"
#include "Metrics/FreqSpectrum.hpp"
#include "PlotCommon.hpp"
#include <QBrush>
#include <QButtonGroup>
#include <QColor>
#include <QCursor>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPen>
#include <QPushButton>
#include <QRadioButton>
#include <QSizePolicy>
#include <QSplitter>
#include <QToolTip>
#include <QVBoxLayout>
#include <QVector>
#include <Qt>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <qcustomplot.h>
#include <span>
#include <uspam/reconParams.hpp>

AScanPlot::AScanPlot(ReconParamsController *reconParams, QWidget *parent)
    : QWidget(parent), m_reconParams(reconParams), customPlot(new CustomPlot),
      m_freqSpectrum(new FreqSpectrum)

{

  /*
   * Setup the AScan customPlot
   */
  {
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");

    // Set minimum size
    customPlot->setMinimumHeight(200); // NOLINT(*-magic-numbers)

    // generate some data
    {
      constexpr int N = 201;

      QVector<double> x(N);
      QVector<double> y(N);
      for (int i = 0; i < N; ++i) {
        constexpr double fct = std::numbers::pi * 2 / N;
        x[i] = i * fct;        // x goes from 0 to 2pi
        y[i] = std::sin(x[i]); // plot sine wave
      }
      plot(x, y, {});
      customPlot->resetZoom();
    }
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

      // FWHM label element
      layout->addWidget(customPlot->FWHMLabel());

      // Horizontal line separator
      {
        auto *line = new QFrame;
        line->setFrameShape(QFrame::Shape::HLine);
        line->setFrameShadow(QFrame::Sunken);
        layout->addWidget(line);
      }

      // Plot Type Selectors
      {
        auto *grid = new QGridLayout;
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

    splitter->addWidget(m_freqSpectrum);

    // Stretchable spacer
    splitter->addWidget(new QWidget);

    // layout->setSpacing(0);
    // layout->setContentsMargins(0, 0, 0, 0);
  }

  this->setSizePolicy(QSizePolicy::MinimumExpanding,
                      QSizePolicy::MinimumExpanding);
  this->setMinimumSize(QSize{500, 500}); // NOLINT(*-magic-numbers)
}

void AScanPlot::showPointToolTip(QMouseEvent *event) {
  const auto pos = event->pos();
  const auto x = customPlot->xAxis->pixelToCoord(pos.x());
  const auto y = customPlot->yAxis->pixelToCoord(pos.y());

  setToolTip(QString("%1, %2").arg(x).arg(y));
}

void AScanPlot::plotCurrentAScan() {
  if (m_data == nullptr) [[unlikely]] {
    return;
  }

  // Correct for flip and rotation in the selected AScan idx
  // and store result in m_AScanPlotIdx
  {
    const bool flip = m_reconParams->params.PA.flip(m_data->frameIdx);
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

  // Reset meta
  CustomPlot::PlotMeta plotMeta;

  switch (m_type) {
  case PlotType::RFRaw: {
    // Original RF
    const auto &rf = m_data->rf;
    const std::span y{rf.colptr(m_AScanPlotIdx), rf.n_rows};
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    customPlot->xAxis2->setLabel("");
    customPlot->xAxis2->setTickLabels(false);

    plotMeta.name = "Raw RF";
    plot(y, plotMeta);
  } break;

  case PlotType::RFBeamformedPA: {
    // PA rfEnv
    plotMeta.xScaler = MM_PER_PIXEL_PA;
    plotMeta.xUnit = "mm";
    plotMeta.name = "Beamformed RF (PA)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    customPlot->xAxis2->setLabel("Depth (mm)");
    customPlot->xAxis2->setTickLabels(true);

    const auto &rf = m_data->PA.rfBeamformed;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, plotMeta);
  } break;

  case PlotType::RFBeamformedUS: {
    // US rfEnv
    plotMeta.xScaler = MM_PER_PIXEL_US;
    plotMeta.xUnit = "mm";
    plotMeta.name = "Beamformed RF (US)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    customPlot->xAxis2->setLabel("Depth (mm)");
    customPlot->xAxis2->setTickLabels(true);

    const auto &rf = m_data->US.rfBeamformed;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, plotMeta);
  } break;

  case PlotType::RFEnvPA: {
    // PA rfEnv
    plotMeta.xScaler = MM_PER_PIXEL_PA;
    plotMeta.xUnit = "mm";
    plotMeta.name = "RF Envelope (PA)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    customPlot->xAxis2->setLabel("Depth (mm)");
    customPlot->xAxis2->setTickLabels(true);

    const auto &rf = m_data->PA.rfEnv;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, plotMeta);
  } break;
  case PlotType::RFEnvUS: {
    // US rfEnv
    plotMeta.xScaler = MM_PER_PIXEL_US;
    plotMeta.xUnit = "mm";
    plotMeta.name = "RF Envelope (US)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    customPlot->xAxis2->setLabel("Depth (mm)");
    customPlot->xAxis2->setTickLabels(true);

    const auto &rf = m_data->US.rfEnv;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, plotMeta);
  } break;

  case PlotType::RFLogPA: {
    // US rfLog
    plotMeta.autoScaleY = false;
    plotMeta.yMax = 0;
    plotMeta.yMax = 256; // NOLINT(*-magic-numbers)
    plotMeta.xScaler = MM_PER_PIXEL_PA;
    plotMeta.xUnit = "mm";
    plotMeta.name = "RF Log (PA)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal");
    customPlot->xAxis2->setLabel("Depth (mm)");
    customPlot->xAxis2->setTickLabels(true);

    const auto &rf = m_data->PA.rfLog;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, plotMeta);
  } break;

  case PlotType::RFLogUS: {
    // US rfLog
    plotMeta.autoScaleY = false;
    plotMeta.yMax = 0;
    plotMeta.yMax = 256; // NOLINT(*-magic-numbers)
    plotMeta.xScaler = MM_PER_PIXEL_US;
    plotMeta.xUnit = "mm";
    plotMeta.name = "RF Log (US)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal");
    customPlot->xAxis2->setLabel("Depth (mm)");
    customPlot->xAxis2->setTickLabels(true);

    const auto &rf = m_data->US.rfLog;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y, plotMeta);
  } break;

  case Size:
    // NOOP
    break;
  }

  if (m_type != m_lastType) {
    customPlot->resetZoom();
    m_lastType = m_type;
  }
}

void AScanPlot::handleAScanSelected(int idx) {
  // The index received here is in canvas pixmap coordinates (i.e. doesn't
  // account for flip and rotation offset)
  m_AScanPlotIdx_canvas = idx;

  plotCurrentAScan();
}