#include "CanvasAnnotationModel.hpp"
#include <cassert>

Annotation::Annotation(Type type, const QList<QPointF> &points,
                       const QColor &color)
    : m_type(type), m_polygon(points), m_color(color) {}

Annotation::Annotation(const QLineF &line, const QColor &color)
    : m_type(Line), m_polygon({line.p1(), line.p2()}), m_color(color) {}

Annotation::Annotation(const QRectF &rect, const QColor &color)
    : m_type(Rect), m_polygon({rect.topLeft(), rect.bottomRight()}),
      m_color(color) {}

auto Annotation::line() const -> QLineF {
  assert(m_polygon.size() == 2);
  return {m_polygon[0], m_polygon[1]};
};

auto Annotation::rect() const -> QRectF {
  assert(m_polygon.size() == 2);
  return {m_polygon[0], m_polygon[1]};
};
