#include "geometryUtils.hpp"
#include <algorithm>

namespace geometry {

QPointF clipPoint(const QPointF &point, const QPointF &minPoint,
                  const QPointF &maxPoint) {
  auto clippedX = std::max(minPoint.x(), std::min(maxPoint.x(), point.x()));
  auto clippedY = std::max(minPoint.y(), std::min(maxPoint.y(), point.y()));
  return {clippedX, clippedY};
}

QPoint clipPoint(const QPoint &point, const QPoint &minPoint,
                 const QPoint &maxPoint) {
  auto clippedX = std::max(minPoint.x(), std::min(maxPoint.x(), point.x()));
  auto clippedY = std::max(minPoint.y(), std::min(maxPoint.y(), point.y()));
  return {clippedX, clippedY};
}

qreal calcMagnitude(const QPointF &pt) {
  return std::sqrt(pt.x() * pt.x() + pt.y() * pt.y());
}

QPointF calcNormalVec(const QLineF &line) {
  const auto vec = line.p2() - line.p1();
  QPointF normal(vec.y(), -vec.x());
  auto magnitude = calcMagnitude(vec);
  return normal / magnitude;
}

QRectF translateBounded(const QRectF &rect, const QPointF &delta,
                        const QRectF &bound) {
  QRectF translated = rect.translated(delta);

  if (translated.left() < bound.left()) {
    translated.moveLeft(bound.left());
  } else if (translated.right() > bound.right()) {
    translated.moveRight(bound.right());
  }

  if (translated.top() < bound.top()) {
    translated.moveTop(bound.top());
  } else if (translated.bottom() > bound.bottom()) {
    translated.moveBottom(bound.bottom());
  }

  return translated;
}
} // namespace geometry