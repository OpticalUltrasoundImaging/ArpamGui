#pragma once
#include "geometryUtils.hpp"
#include <QColor>
#include <QLineF>
#include <QPolygonF>
#include <QRectF>
#include <QString>
#include <array>

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
class Annotation {
public:
  enum Type { Line, Rect, Fan, Polygon, Size };
  inline const static std::array TypeToString = {
      QString("Line"), QString("Rect"), QString("Fan"), QString("Polygon")};

  /* Constructors */
  Annotation(Type type, const QList<QPointF> &points, const QColor &color,
             QString name = {});
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

  [[nodiscard]] auto type() const { return m_type; }
  void setType(Type type) { m_type = type; }

  // For Line, the 2 points are {p1, p2}
  [[nodiscard]] auto line() const -> QLineF {
    assert(m_polygon.size() == 2);
    return {m_polygon[0], m_polygon[1]};
  };

  // For Rect, the 2 points are {top_left, bottom_right}
  [[nodiscard]] auto rect() const -> QRectF {
    assert(m_polygon.size() >= 2);
    return {m_polygon[0], m_polygon[1]};
  };

  // For arc, the 3rd point stores the startAngle (x) and spanAngle (y)
  [[nodiscard]] auto arc() const -> Arc {
    assert(m_polygon.size() == 3);
    const auto pt = m_polygon[2];
    return Arc{pt.x(), pt.y()};
  }

  [[nodiscard]] auto polygon() const -> QPolygonF { return m_polygon; };
  void setPolygon(const QPolygonF &polygon) { m_polygon = polygon; }

  [[nodiscard]] auto color() const -> QColor { return m_color; }
  void setColor(QColor color) { m_color = color; }

  [[nodiscard]] auto name() const { return m_name; }
  void setName(QString name) { m_name = std::move(name); }

  static QString typeToString(Type type) { return TypeToString.at(type); }

private:
  Type m_type;
  QPolygonF m_polygon;
  QColor m_color;
  QString m_name;
};

} // namespace annotation