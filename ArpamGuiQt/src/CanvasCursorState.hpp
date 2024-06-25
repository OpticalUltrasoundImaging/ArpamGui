#pragma once

#include <QLineF>
#include <QPointF>
#include <QRectF>
#include <algorithm>
#include <geometryUtils.hpp>
#include <tuple>

struct CanvasCursorState {
  bool leftButtonDown = false;
  bool middleButtonDown = false;
  bool rightButtonDown = false;
  QPointF pos; // current cursor position in pixmap coord
  QPointF startPos;

  // For Fan drawing
  double lastAngle{};
  double angleOffset{};

  [[nodiscard]] auto line() const { return QLineF(startPos, pos); }

  [[nodiscard]] auto rect() const {
    qreal x = qMin(pos.x(), startPos.x());
    qreal y = qMin(pos.y(), startPos.y());

    qreal w = qAbs(pos.x() - startPos.x());
    qreal h = qAbs(pos.y() - startPos.y());
    return QRectF(x, y, w, h);
  }

  // Get current cursor position in the circular BScan's angle in degrees given
  // the bounding rect. This angle starts at the "positive X axis" or 3 O'clock
  // position and goes clockwise.
  [[nodiscard]] inline auto angleDeg(const QRectF &rect) const {
    return geometry::calcAngleFromPos(rect, pos);
  }

  // Get the AScan under the cursor position in the circular BScan.
  [[nodiscard]] inline auto
  selectAScan(const QRectF &rect, const int numAScansPerBScan = 1000) const {
    // Convert the angle to start at 12 O'clock
    constexpr double degOffset = -90;
    auto angle = angleDeg(rect) + degOffset;
    if (angle > 360) {
      angle -= 360;
    } else if (angle < 0) {
      angle += 360;
    }

    const auto idx = std::clamp(
        static_cast<int>(std::round(angle / 360 * numAScansPerBScan)), 0,
        numAScansPerBScan - 1);

    // Get the line in pixmap space representing the AScan
    const auto center = rect.center();
    const auto diff = pos - center;
    const auto len = geometry::calcMagnitude(diff);
    // Images have a radius of 500 pixels
    const auto fct = 500 / len;

    const QPointF p2(center.x() + fct * (pos.x() - center.x()),
                     center.y() + fct * (pos.y() - center.y()));

    const QLineF line(center, p2);

    return std::tuple{idx, line};
  }
};
