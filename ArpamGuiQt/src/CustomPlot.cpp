#include "CustomPlot.hpp"

CustomPlot::CustomPlot(QWidget *parent) : QCustomPlot(parent) {
  setMouseTracking(true);

  // Mouse crosshair
  connect(this, &QCustomPlot::mouseMove, [this](QMouseEvent *event) {
    const auto x = xAxis->pixelToCoord(event->pos().x());
    const auto y = yAxis->pixelToCoord(event->pos().y());

    const auto txt = QString("%1 , %2").arg(x).arg(y, 6, 'f', 4);
    QToolTip::showText(QCursor::pos(), txt);
  });
}

void CustomPlot::leaveEvent(QEvent *event) {
  QCustomPlot::leaveEvent(event);
  emit mouseLeft();
}
