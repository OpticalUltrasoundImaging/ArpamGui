#include "CustomPlot.hpp"
#include "Metrics/FWHMTracer.hpp"
#include "PlotCommon.hpp"
#include <QAction>
#include <qcustomplot.h>
#include <qnamespace.h>

CustomPlot::CustomPlot(QWidget *parent)
    : QCustomPlot(parent), m_FWHMtracers(new FWHMTracers(this)),
      m_FWHMLabel(new QLabel) {

  // Appearance
  {
    // Set dark background color
    setBackground(QColor(0, 0, 0)); // Black

    // Style axis
    const auto styleAxis = [](QCPAxis *axis) {
      axis->setBasePen(QPen(Qt::white));
      axis->setTickPen(QPen(Qt::white));
      axis->setSubTickPen(QPen(Qt::white));
      axis->setTickLabelColor(Qt::white);
      axis->setLabelColor(Qt::white);
    };
    styleAxis(xAxis);
    styleAxis(xAxis2);
    styleAxis(yAxis);
    styleAxis(yAxis2);

    xAxis2->setVisible(true);
    yAxis2->setVisible(true);

    // customPlot->xAxis->grid()->setVisible(false);
    // customPlot->yAxis->grid()->setVisible(false);

    // NOLINTBEGIN(*-magic-numbers)
    legend->setBrush(QBrush(QColor(255, 255, 255, 100)));
    legend->setVisible(true);

    setupAxis(xAxis, {}, true, 0, TICK_LENGTH, 0, SUBTICK_LENGTH);
    setupAxis(yAxis, {}, true, 0, TICK_LENGTH, 0, SUBTICK_LENGTH);
    setupAxis(xAxis2, {}, false);
    setupAxis(yAxis2, {}, false);

    setMinimumHeight(200);
    // NOLINTEND(*-magic-numbers)
  }

  // Interactions
  {
    // customPlot->setMouseTracking(true);

    // This line fails on MSVC for some reason
    // connect(customPlot, &QCustomPlot::mouseMove, this,
    //         [this](QMouseEvent *event) {
    //           const auto x =
    //           customPlot->xAxis->pixelToCoord(event->pos().x()); const
    //           auto y = customPlot->yAxis->pixelToCoord(event->pos().y());

    //           const auto txt = QString("%1 , %2").arg(x).arg(y, 6, 'f',
    //           4); QToolTip::showText(QCursor::pos(), txt);
    //         });

    setInteraction(QCP::iRangeZoom, true);
    setInteraction(QCP::iRangeDrag, true);
    setInteraction(QCP::iSelectItems, true);
    setInteraction(QCP::iSelectPlottables, true);
    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
  }

  /*
   *Actions
   */

  // Reset Zoom
  {
    m_actResetZoom = new QAction("Reset Zoom");
    connect(m_actResetZoom, &QAction::triggered, [this] {
      xAxis->setRange(m_meta.xMin, m_meta.xMax);
      yAxis->setRange(m_meta.yMin, m_meta.yMax);
      replot();
    });
    addAction(m_actResetZoom);
  }

  // Show FWHM
  {
    m_actShowFWHM = new QAction("Show FWHM");
    m_actShowFWHM->setCheckable(true);
    m_actShowFWHM->setChecked(true);
    connect(m_actShowFWHM, &QAction::triggered, [this](bool checked) {
      m_FWHMtracers->toggle();
      replot();
    });
    addAction(m_actShowFWHM);
  }

  // Show Legend
  {
    m_actShowLegend = new QAction("Show Legend");
    m_actShowLegend->setCheckable(true);
    m_actShowLegend->setChecked(true);
    connect(m_actShowLegend, &QAction::triggered, [this](bool checked) {
      legend->setVisible(checked);
      replot();
    });
    addAction(m_actShowLegend);
  }

  setContextMenuPolicy(Qt::ContextMenuPolicy::ActionsContextMenu);

  // Add graph 0
  {
    addGraph();
    graph(0)->setPen(QPen(Qt::green));
  }
}

void CustomPlot::plot(const QVector<double> &x, const QVector<double> &y,
                      const CustomPlot::PlotMeta &meta) {
  assert(x.size() == y.size());
  m_meta = meta;

  // Compute FWHM
  const auto fwhm = m_FWHMtracers->updateData(x, y);
  // FWHM width in X
  const auto width = fwhm.width();

  // FWHM label
  if (!meta.xUnit.isEmpty()) {
    const auto xWidth = width * meta.xScaler;
    m_FWHMLabel->setText(QString("FWHM: %1 samples, %2 %3")
                             .arg(width)
                             .arg(xWidth)
                             .arg(meta.xUnit));

  } else {
    m_FWHMLabel->setText(QString("FWHM: %1 samples").arg(width));
  }

  // Title
  graph(0)->setName(meta.name);

  // replot
  graph(0)->setData(x, y, true);

  // x range
  m_meta.xMin = x.front();
  m_meta.xMax = x.back();
  xAxis->setRange(m_meta.xMin, m_meta.xMax);
  xAxis2->setRange(m_meta.xMin * m_meta.xScaler, m_meta.xMax * m_meta.xScaler);

  // y range
  if (m_meta.autoScaleY) {
    const auto [min, max] = std::minmax_element(y.cbegin(), y.cend());
    m_meta.yMin = *min;
    m_meta.yMax = *max;
  }
  yAxis->setRange(m_meta.yMin, m_meta.yMax);

  replot();
}

void CustomPlot::ensureX(int size) {
  if (m_x.size() != size) {
    m_x.resize(size);
    std::iota(m_x.begin(), m_x.end(), 0);
  }
}
