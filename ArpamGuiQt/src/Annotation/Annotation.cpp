#include "Annotation/Annotation.hpp"
#include "jsonUtils.hpp"
#include "strConvUtils.hpp"
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>

namespace {

auto deserializeListOfPoints(const rapidjson::Value &jsonArray) {
  QList<QPointF> points;
  for (const auto &p : jsonArray.GetArray()) {
    const auto &pp = p.GetArray();
    QPointF point{pp[0].GetDouble(), pp[1].GetDouble()};
    points.append(point);
  }
  return points;
}

} // namespace

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

rapidjson::Value Annotation::serializeToJson(
    rapidjson::Document::AllocatorType &allocator) const {
  rapidjson::Value obj(rapidjson::kObjectType);

  obj.AddMember("type",
                jsonUtils::serializeString(typeToString(m_type), allocator),
                allocator);

  obj.AddMember("points",
                jsonUtils::serializeListOfPoints(m_polygon, allocator),
                allocator);

  obj.AddMember(
      "color",
      jsonUtils::serializeString(m_color.name(QColor::HexRgb), allocator),
      allocator);

  obj.AddMember("name", jsonUtils::serializeString(m_name, allocator),
                allocator);

  return obj;
}

} // namespace annotation