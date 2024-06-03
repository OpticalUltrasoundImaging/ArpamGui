#pragma once

#include <QLineF>
#include <QPointF>
#include <QRectF>

struct CanvasCursorState {
  bool leftButtonDown = false;
  bool middleButtonDown = false;
  bool rightButtonDown = false;
  QPointF currPos; // current cursor position in scaled pixmap coord
  QPointF startPos;

  QPointF currPosOrigal; // cursor position in original pixmap coord

  [[nodiscard]] auto getLine() const { return QLineF(startPos, currPos); }
  [[nodiscard]] auto getRect() const {
    qreal x = qMin(currPos.x(), startPos.x());
    qreal y = qMin(currPos.y(), startPos.y());

    qreal w = qAbs(currPos.x() - startPos.x());
    qreal h = qAbs(currPos.y() - startPos.y());
    return QRectF(x, y, w, h);
  }
};
