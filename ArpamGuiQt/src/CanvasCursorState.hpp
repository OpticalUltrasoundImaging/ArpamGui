#pragma once

#include <QLineF>
#include <QPointF>
#include <QRectF>
#include <geometryUtils.hpp>

struct CanvasCursorState {
  bool leftButtonDown = false;
  bool middleButtonDown = false;
  bool rightButtonDown = false;
  QPointF pos; // current cursor position in pixmap coord
  QPointF startPos;

  [[nodiscard]] auto line() const { return QLineF(startPos, pos); }

  [[nodiscard]] auto rect() const {
    qreal x = qMin(pos.x(), startPos.x());
    qreal y = qMin(pos.y(), startPos.y());

    qreal w = qAbs(pos.x() - startPos.x());
    qreal h = qAbs(pos.y() - startPos.y());
    return QRectF(x, y, w, h);
  }

  [[nodiscard]] auto angleDeg(const QRectF &rect) const {
    return geometry::calcAngleFromPos(rect, pos);
  }
};
