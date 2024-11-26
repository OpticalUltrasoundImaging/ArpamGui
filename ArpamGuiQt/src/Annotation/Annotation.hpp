#pragma once

#include "geometryUtils.hpp"
#include <QColor>
#include <QLineF>
#include <QList>
#include <QPolygonF>
#include <QRectF>
#include <QString>
#include <algorithm>
#include <array>
#include <cmath>
#include <rapidjson/document.h>

namespace annotation {

/**
Pair of angles (each 0-360) that denote a fan shape centered at the center of
the square image.

Draws the arc defined by the given rectangle, startAngle and spanAngle.
The startAngle and spanAngle are in degrees
Positive values for the angles mean counter-clockwise
while negative values mean the clockwise direction.
Zero degrees is at the 3 o'clock position.

https://doc.qt.io/qt-6/qpainterpath.html#arcTo
*/
struct Arc {
  double startAngle;
  double spanAngle;

  [[nodiscard]] inline auto startAngleRadians() const {
    return geometry::deg2rad(startAngle);
  }

  [[nodiscard]] inline auto spanAngleRadians() const {
    return geometry::deg2rad(spanAngle);
  }
};

// Points are stored in a QPolygonF, which is just a QList<QPointF>
// https://doc.qt.io/qt-6/qpolygonf.html#details
struct Annotation {
  enum Type { Line, Rect, Fan, Polygon, Size };

  Type type{Line};
  QPolygonF polygon;
  QColor color;
  QString name;

  inline const static std::array<QString, Size> TypeToString = {
      QString("Line"), QString("Rect"), QString("Fan"), QString("Polygon")};

  /* Constructors */
  Annotation() = default;
  /* Copy/move constructor/assignment */
  Annotation(const Annotation &other) = default;
  Annotation &operator=(const Annotation &other) = default;
  Annotation(Annotation &&) = default;
  Annotation &operator=(Annotation &&) = default;

  /*
   * For line, the polygon contains 2 points: {p1, p2}
   */
  Annotation(const QLineF &line, const QColor &color, QString name = {});

  /*
   * For rect, the polygon contains 2 points: {topLeft, bottomRight}
   */
  Annotation(const QRectF &rect, const QColor &color, QString name = {});

  /*
   * For arc, the Polygon contains 3 points:
   * {rect.topLeft, rect.bottomRight, {startAngle, spanAngle}}
   */
  Annotation(const Arc &arc, const QRectF &rect, const QColor &color,
             QString name = {});

  /*
   *  For polygon
   */
  Annotation(QPolygonF polygon, const QColor &color, QString name = {});

  ~Annotation() = default;

  // For Line, the 2 points are {p1, p2}
  [[nodiscard]] auto line() const -> QLineF {
    assert(type == Line);
    assert(polygon.size() == 2);
    return {polygon[0], polygon[1]};
  };
  void setLine(const QLineF &line) {
    assert(type == Line);
    polygon = {line.p1(), line.p2()};
  }

  // For Rect, the 2 points are {top_left, bottom_right}
  [[nodiscard]] auto rect() const -> QRectF {
    assert(type == Line || type == Rect || type == Fan); // NOLINT
    assert(polygon.size() >= 2);
    return QRectF{polygon[0], polygon[1]}.normalized();
  };
  void setRect(const QRectF &rect) {
    assert(type == Rect || type == Fan); // NOLINT
    assert(polygon.size() >= 2);
    polygon[0] = rect.topLeft();
    polygon[1] = rect.bottomRight();
  }

  // For arc, the 3rd point stores the startAngle (x) and spanAngle (y)
  [[nodiscard]] auto arc() const -> Arc {
    assert(type == Fan);
    assert(polygon.size() == 3);
    const auto pt = polygon[2];
    return Arc{pt.x(), pt.y()};
  }
  void setArc(const Arc &arc, const QRectF &rect) {
    assert(type == Fan);
    polygon = {rect.topLeft(), rect.bottomRight(),
               QPointF{arc.startAngle, arc.spanAngle}};
  }
  void setArc(const Arc &arc) {
    assert(type == Fan);
    assert(polygon.size() == 3);
    polygon[2] = QPointF{arc.startAngle, arc.spanAngle};
  }
  [[nodiscard]] auto startAngle() const {
    assert(type == Fan);
    assert(polygon.size() == 3);
    return polygon[2].x();
  }
  void setStartAngle(double startAngle) {
    assert(type == Fan);
    assert(polygon.size() == 3);
    polygon[2].setX(startAngle);
  }
  [[nodiscard]] auto spanAngle() const {
    assert(type == Fan);
    assert(polygon.size() == 3);
    return polygon[2].y();
  }
  void setSpanAngle(double spanAngle) {
    assert(type == Fan);
    assert(polygon.size() == 3);
    polygon[2].setY(spanAngle);
  }

  static const QString &typeToString(Type type) {
    assert(type < Type::Size);
    return TypeToString.at(type);
  }

  [[nodiscard]] bool tooSmall() const {
    constexpr int rectThresh = 20;
    constexpr int angleThresh = 2;
    switch (type) {
    case Line:
    case Rect: {
      const auto r = rect();
      const auto span = r.bottomRight() - r.topLeft();
      return std::abs(span.x() + span.y()) <= rectThresh;
    } break;
    case Fan: {
      return std::abs(arc().spanAngle) <= angleThresh;
    }
    case Polygon: {
      const auto r = polygon.boundingRect();
    } break;
    case Size:
    default:
      break;
    }
    return true;
  }

  static Type typeFromString(const QString &type) {
    // NOLINTNEXTLINE(*-auto)
    const auto it = std::find(TypeToString.cbegin(), TypeToString.cend(), type);
    if (it != TypeToString.cend()) {
      return static_cast<Type>(std::distance(TypeToString.cbegin(), it));
    }
    return static_cast<Type>(0);
  }

  // Json serialization
  [[nodiscard]] rapidjson::Value
  serializeToJson(rapidjson::Document::AllocatorType &allocator) const;
  [[nodiscard]] static Annotation
  deserializeFromJson(const rapidjson::Value &value);
};

} // namespace annotation
