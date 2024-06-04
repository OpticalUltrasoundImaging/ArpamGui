#pragma once

#include <QLineF>
#include <QPointF>
#include <QRectF>

struct CanvasCursorState {
  bool leftButtonDown = false;
  bool middleButtonDown = false;
  bool rightButtonDown = false;
  QPointF pos; // current cursor position in pixmap coord
  QPointF startPos;

  [[nodiscard]] auto getLine() const { return QLineF(startPos, pos); }
  [[nodiscard]] auto getRect() const {
    qreal x = qMin(pos.x(), startPos.x());
    qreal y = qMin(pos.y(), startPos.y());

    qreal w = qAbs(pos.x() - startPos.x());
    qreal h = qAbs(pos.y() - startPos.y());
    return QRectF(x, y, w, h);
  }
};
