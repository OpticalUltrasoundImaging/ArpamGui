#include "AScanPlot.hpp"
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

void setupAxis(QCPAxis *axis, const QString &label = {}, bool tickLabels = true,
               int tickInside = 0, int tickOutside = 0, int subTickInside = 0,
               int subTickOutside = 0) {
  axis->setLabel(label);
  axis->setTickLabels(tickLabels);
  axis->setTickLength(tickInside, tickOutside);
  axis->setSubTickLength(subTickInside, subTickOutside);
}

template <typename T>
FWHM<T> AScanFWHMTracers::updateData(const QVector<T> &x, const QVector<T> &y,
                                     int graphIdx) {
  const auto fwhm = calcFWHM<T>(x, y);

  peakTracer->setGraph(customPlot->graph(graphIdx));
  peakTracer->setGraphKey(x[fwhm.peakIdx]);

  lineLower->start->setCoords(fwhm.lowerX, 0);
  lineLower->end->setCoords(fwhm.lowerX, y[fwhm.lowerIdx]);

  lineUpper->start->setCoords(fwhm.upperX, 0);
  lineUpper->end->setCoords(fwhm.upperX, y[fwhm.upperIdx]);
  return fwhm;
}

AScanPlot::AScanPlot(ReconParamsController *reconParams, QWidget *parent)
    : QWidget(parent), m_reconParams(reconParams), customPlot(new QCustomPlot),
      m_FWHMtracers(customPlot), m_FWHMLabel(new QLabel) {

  /*
   * Setup the customPlot
   */
  {
    // Interaction
    customPlot->setMouseTracking(true);
    // This line fails on MSVC for some reason
    // connect(customPlot, &QCustomPlot::mouseMove, this,
    //         [this](QMouseEvent *event) {
    //           const auto x =
    //           customPlot->xAxis->pixelToCoord(event->pos().x()); const auto y
    //           = customPlot->yAxis->pixelToCoord(event->pos().y());

    //           const auto txt = QString("%1 , %2").arg(x).arg(y, 6, 'f', 4);
    //           QToolTip::showText(QCursor::pos(), txt);
    //         });

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

    // NOLINTNEXTLINE(*-magic-numbers)
    customPlot->legend->setBrush(QBrush(QColor(255, 255, 255, 100)));
    customPlot->legend->setVisible(true);

    // create graph and assign data to it:
    customPlot->addGraph();
    customPlot->graph(0)->setPen(QPen(Qt::green));

    // Remove grid
    customPlot->xAxis->grid()->setVisible(false);
    customPlot->yAxis->grid()->setVisible(false);

    // Add top and right axis to have top and left border
    customPlot->xAxis2->setVisible(true);
    customPlot->yAxis2->setVisible(true);

    // NOLINTBEGIN(*-magic-numbers)
    setupAxis(customPlot->xAxis, "Samples", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);
    setupAxis(customPlot->yAxis, "Signal (V)", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);
    setupAxis(customPlot->xAxis2, {}, false);
    setupAxis(customPlot->yAxis2, {}, false);
    // NOLINTEND(*-magic-numbers)

    // Interaction
    customPlot->setInteraction(QCP::iRangeZoom, true);
    customPlot->setInteraction(QCP::iRangeDrag, true);
    customPlot->setInteraction(QCP::iSelectItems, true);
    customPlot->setInteraction(QCP::iSelectPlottables, true);

    // Set minimum size
    customPlot->setMinimumHeight(200); // NOLINT(*-magic-numbers)

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

      // Horizontal control buttons
      {
        auto *hlayout = new QHBoxLayout;
        layout->addLayout(hlayout);

        {
          auto *btn = new QPushButton("Reset zoom");
          hlayout->addWidget(btn);

          connect(btn, &QPushButton::clicked, [this] {
            customPlot->xAxis->setRange(m_plotMeta.xMin, m_plotMeta.xMax);
            customPlot->yAxis->setRange(m_plotMeta.yMin, m_plotMeta.yMax);
            customPlot->replot();
          });
        }

        {
          // Button to show/hide FWHM display on the plot
          auto *btn = new QPushButton("Show FWHM");
          btn->setCheckable(true);
          btn->setChecked(true); // Show by default
          connect(btn, &QPushButton::clicked, [this](bool checked) {
            m_FWHMtracers.toggle();
            customPlot->replot();
          });
          hlayout->addWidget(btn);
        }

        {
          // Button to show/hide legend
          auto *btn = new QPushButton("Show Legend");
          btn->setCheckable(true);
          btn->setChecked(true); // Show by default
          connect(btn, &QPushButton::clicked, [this](bool checked) {
            customPlot->legend->setVisible(checked);
            customPlot->replot();
          });
          hlayout->addWidget(btn);
        }
      }

      {
        // FWHM label element
        layout->addWidget(m_FWHMLabel);
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
  this->setMinimumSize(QSize{500, 500}); // NOLINT(*-magic-numbers)
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

template <typename T> void AScanPlot::plot(std::span<const T> y) {
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
  plot(m_x, m_y);
}

void AScanPlot::plot(const QVector<FloatType> &x, const QVector<FloatType> &y) {
  assert(x.size() == y.size());

  // Compute FWHM
  const auto fwhm = m_FWHMtracers.updateData(x, y);
  // FWHM width in X
  const auto width = fwhm.width();

  // FWHM label
  if (!m_plotMeta.xUnit.isEmpty()) {
    const auto xWidth = width * m_plotMeta.xScaler;
    m_FWHMLabel->setText(QString("FWHM: %1 samples, %2 %3")
                             .arg(width)
                             .arg(xWidth)
                             .arg(m_plotMeta.xUnit));

  } else {
    m_FWHMLabel->setText(QString("FWHM: %1 samples").arg(width));
  }

  // Title
  customPlot->graph(0)->setName(m_plotMeta.name);

  // replot
  customPlot->graph(0)->setData(x, y, true);

  // x range
  m_plotMeta.xMin = x.front();
  m_plotMeta.xMax = x.back();
  customPlot->xAxis->setRange(m_plotMeta.xMin, m_plotMeta.xMax);
  customPlot->xAxis2->setRange(m_plotMeta.xMin * m_plotMeta.xScaler,
                               m_plotMeta.xMax * m_plotMeta.xScaler);

  // y range
  if (m_plotMeta.autoScaleY) {
    const auto [min, max] = std::minmax_element(x.cbegin(), y.cend());
    m_plotMeta.yMin = *min;
    m_plotMeta.yMax = *max;
  }
  customPlot->yAxis->setRange(m_plotMeta.yMin, m_plotMeta.yMax);

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

  // Reset meta
  m_plotMeta = PlotMeta{};

  switch (m_type) {
  case PlotType::RFRaw: {
    // Original RF
    const auto &rf = m_data->rf;
    const std::span y{rf.colptr(m_AScanPlotIdx), rf.n_rows};
    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    setupAxis(customPlot->xAxis2, {}, false);

    m_plotMeta.name = "Raw RF";
    plot(y);
  } break;

  case PlotType::RFBeamformedPA: {
    // PA rfEnv
    m_plotMeta.xScaler = MM_PER_PIXEL_PA;
    m_plotMeta.xUnit = "mm";
    m_plotMeta.name = "Beamformed RF (PA)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    setupAxis(customPlot->xAxis2, "Depth (mm)", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);

    const auto &rf = m_data->PA.rfBeamformed;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;

  case PlotType::RFBeamformedUS: {
    // US rfEnv
    m_plotMeta.xScaler = MM_PER_PIXEL_US;
    m_plotMeta.xUnit = "mm";
    m_plotMeta.name = "Beamformed RF (US)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    setupAxis(customPlot->xAxis2, "Depth (mm)", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);

    const auto &rf = m_data->US.rfBeamformed;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;

  case PlotType::RFEnvPA: {
    // PA rfEnv
    m_plotMeta.xScaler = MM_PER_PIXEL_PA;
    m_plotMeta.xUnit = "mm";
    m_plotMeta.name = "RF Envelope (PA)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    setupAxis(customPlot->xAxis2, "Depth (mm)", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);

    const auto &rf = m_data->PA.rfEnv;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;
  case PlotType::RFEnvUS: {
    // US rfEnv
    m_plotMeta.xScaler = MM_PER_PIXEL_US;
    m_plotMeta.xUnit = "mm";
    m_plotMeta.name = "RF Envelope (US)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal (V)");
    setupAxis(customPlot->xAxis2, "Depth (mm)", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);

    const auto &rf = m_data->US.rfEnv;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;

  case PlotType::RFLogPA: {
    // US rfLog
    m_plotMeta.autoScaleY = false;
    m_plotMeta.yMax = 0;
    m_plotMeta.yMax = 256; // NOLINT(*-magic-numbers)
    m_plotMeta.xScaler = MM_PER_PIXEL_PA;
    m_plotMeta.xUnit = "mm";
    m_plotMeta.name = "RF Log (PA)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal");
    setupAxis(customPlot->xAxis2, "Depth (mm)", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);

    const auto &rf = m_data->PA.rfLog;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
  } break;

  case PlotType::RFLogUS: {
    // US rfLog
    m_plotMeta.autoScaleY = false;
    m_plotMeta.yMax = 0;
    m_plotMeta.yMax = 256; // NOLINT(*-magic-numbers)
    m_plotMeta.xScaler = MM_PER_PIXEL_US;
    m_plotMeta.xUnit = "mm";
    m_plotMeta.name = "RF Log (US)";

    customPlot->xAxis->setLabel("Samples");
    customPlot->yAxis->setLabel("Signal");
    setupAxis(customPlot->xAxis2, "Depth (mm)", true, 0, TICK_LENGTH, 0,
              SUBTICK_LENGTH);

    const auto &rf = m_data->US.rfLog;
    const std::span y{rf.colptr(m_AScanPlotIdx_canvas), rf.n_rows};
    plot(y);
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