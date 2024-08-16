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
#include <rapidjson/document.h>

namespace annotation {

/**
Pair of angles (each 0-360) that denote a fan shape center at the center of
the square image.

Draws the arc defined by the given rectangle, startAngle and spanAngle.
The startAngle and spanAngle are in degrees
Positive values for the angles mean counter-clockwise
while negative values mean the clockwise direction.
Zero degrees is at the 3 o'clock position.
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
  Annotation(const QLineF &line, const QColor &color, QString name = {});
  Annotation(const QRectF &rect, const QColor &color, QString name = {});
  /* For arc, the Polygon contains 3 points that store
   * {rect.topLeft, rect.bottomRight, {startAngle, spanAngle}}
   */
  Annotation(const Arc &arc, const QRectF &rect, const QColor &color,
             QString name = {});

  /* Copy constructor */
  Annotation(const Annotation &other) = default;
  /* Copy assignment */
  Annotation &operator=(const Annotation &other) = default;
  /* Move constructor */
  Annotation(Annotation &&) = default;
  /* Move assignment */
  Annotation &operator=(Annotation &&) = default;

  ~Annotation() = default;

  // For Line, the 2 points are {p1, p2}
  [[nodiscard]] auto line() const -> QLineF {
    assert(polygon.size() == 2);
    return {polygon[0], polygon[1]};
  };

  // For Rect, the 2 points are {top_left, bottom_right}
  [[nodiscard]] auto rect() const -> QRectF {
    assert(polygon.size() >= 2);
    return {polygon[0], polygon[1]};
  };

  // For arc, the 3rd point stores the startAngle (x) and spanAngle (y)
  [[nodiscard]] auto arc() const -> Arc {
    assert(polygon.size() == 3);
    const auto pt = polygon[2];
    return Arc{pt.x(), pt.y()};
  }

  static QString typeToString(Type type) {
    if (type < Type::Size) {
      return TypeToString.at(type);
    }
    return {};
  }

  static Type typeFromString(const QString &type) {
    // NOLINTNEXTLINE(*-auto)
    const auto it = std::find(TypeToString.cbegin(), TypeToString.cend(), type);
    if (it != TypeToString.cend()) {
      return static_cast<Type>(std::distance(TypeToString.cbegin(), it));
    }
    return static_cast<Type>(0);
  }

  [[nodiscard]] rapidjson::Value
  serializeToJson(rapidjson::Document::AllocatorType &allocator) const;
  [[nodiscard]] static Annotation
  deserializeFromJson(const rapidjson::Value &value);
};

} // namespace annotation
