#pragma once

#include <QLine>
#include <QPoint>

namespace geometry {

// Clip point to minPoint and maxPoint
QPointF clipPoint(const QPointF &point, const QPointF &minPoint,
                  const QPointF &maxPoint);

QPoint clipPoint(const QPoint &point, const QPoint &minPoint,
                 const QPoint &maxPoint);

qreal calcMagnitude(const QPointF &pt);

QPointF calcNormalVec(const QLineF &line);

} // namespace geometry