#pragma once

#include <QLine>
#include <QPoint>
#include <QRectF>
#include <cmath>
#include <tuple>

namespace geometry {

// Clip point to minPoint and maxPoint
[[nodiscard]] QPointF clipPoint(const QPointF &point, const QPointF &minPoint,
                                const QPointF &maxPoint);

[[nodiscard]] QPoint clipPoint(const QPoint &point, const QPoint &minPoint,
                               const QPoint &maxPoint);

[[nodiscard]] qreal calcMagnitude(const QPointF &pt);

[[nodiscard]] QPointF calcNormalVec(const QLineF &line);

[[nodiscard]] QRectF translateBounded(const QRectF &rect, const QPointF &delta,
                                      const QRectF &bound);

// Arc
[[nodiscard]] inline auto rad2deg(double rad) { return rad * 180.0 / M_PI; }
[[nodiscard]] inline auto deg2rad(double deg) { return deg / 180.0 * M_PI; }

// Calculate the angle (in degrees) of a point with respect to a rect
[[nodiscard]] inline double calcAngleFromPos(const QRectF &rect, QPointF pos) {
  const auto center = rect.center();
  // The angle computed by atan2 uses signs to determine the quadrant.
  // y axis is flipped
  auto angle = rad2deg(atan2(center.y() - pos.y(), pos.x() - center.x()));
  if (angle < 0) {
    angle = 360.0 + angle;
  }
  return angle;
}

// Calculate the position on the circle's circumference given a bounding rect
// and an angle (in 1/16 of a degree)
[[nodiscard]] inline QPointF calcPosFromAngle(const QRectF &rect,
                                              double angle) {
  const auto center = rect.center();
  const auto cx = center.x();
  const auto cy = center.y();
  const auto w = rect.width();
  const auto h = rect.height();

  const double startRadians = deg2rad(angle);

  QPointF point(cx + w / 2 * cos(startRadians), cy - h / 2 * sin(startRadians));
  return point;
}

[[nodiscard]] inline std::tuple<QPointF, QPointF>
calcPosFromArc(const QRectF &rect, double startAngleDeg, double spanAngleDeg) {
  // Draw the lines connecting the ends of the arc to the center
  const auto center = rect.center();
  const auto cx = center.x();
  const auto cy = center.y();
  const auto w = rect.width();
  const auto h = rect.height();

  // Calculate the end points of the arc
  const double startRadians = deg2rad(startAngleDeg);
  const double endRadians = deg2rad(startAngleDeg + spanAngleDeg);

  QPointF startPoint(cx + w / 2 * cos(startRadians),
                     cy - h / 2 * sin(startRadians));
  QPointF endPoint(cx + w / 2 * cos(endRadians), cy - h / 2 * sin(endRadians));
  return {startPoint, endPoint};
}

} // namespace geometry