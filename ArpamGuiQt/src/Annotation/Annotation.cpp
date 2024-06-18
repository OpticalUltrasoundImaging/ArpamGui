#include "Annotation/Annotation.hpp"

namespace annotation {

Annotation::Annotation(Type type, const QList<QPointF> &points,
                       const QColor &color, QString name)
    : m_type(type), m_polygon(points), m_color(color), m_name(std::move(name)) {
}

Annotation::Annotation(const QLineF &line, const QColor &color, QString name)
    : m_type(Line), m_polygon({line.p1(), line.p2()}), m_color(color),
      m_name(std::move(name)) {}

Annotation::Annotation(const QRectF &rect, const QColor &color, QString name)
    : m_type(Rect), m_polygon({rect.topLeft(), rect.bottomRight()}),
      m_color(color), m_name(std::move(name)) {}

Annotation::Annotation(const Arc &arc, const QRectF &rect, const QColor &color,
                       QString name)
    : m_type(Fan), m_polygon({rect.topLeft(), rect.bottomRight(),
                              QPointF{arc.startAngle, arc.spanAngle}}),
      m_color(color), m_name(std::move(name)) {}

} // namespace annotation