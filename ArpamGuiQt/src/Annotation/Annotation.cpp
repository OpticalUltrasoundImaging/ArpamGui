#include "Annotation/Annotation.hpp"
#include "jsonUtils.hpp"
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

Annotation::Annotation(const QLineF &line, const QColor &color, QString name)
    : polygon({line.p1(), line.p2()}), color(color), name(std::move(name)) {}

Annotation::Annotation(const QRectF &rect, const QColor &color, QString name)
    : type(Rect), polygon({rect.topLeft(), rect.bottomRight()}), color(color),
      name(std::move(name)) {}

Annotation::Annotation(const Arc &arc, const QRectF &rect, const QColor &color,
                       QString name)
    : type(Fan), polygon({rect.topLeft(), rect.bottomRight(),
                          QPointF{arc.startAngle, arc.spanAngle}}),
      color(color), name(std::move(name)) {}

rapidjson::Value Annotation::serializeToJson(
    rapidjson::Document::AllocatorType &allocator) const {
  rapidjson::Value obj(rapidjson::kObjectType);

  obj.AddMember("type",
                jsonUtils::serializeString(typeToString(type), allocator),
                allocator);

  obj.AddMember("points", jsonUtils::serializeListOfPoints(polygon, allocator),
                allocator);

  obj.AddMember(
      "color",
      jsonUtils::serializeString(color.name(QColor::HexRgb), allocator),
      allocator);

  obj.AddMember("name", jsonUtils::serializeString(name, allocator), allocator);

  return obj;
}

Annotation Annotation::deserializeFromJson(const rapidjson::Value &value) {
  Annotation anno;

  if (const auto it = value.FindMember("type"); it != value.MemberEnd()) {
    anno.type = typeFromString(it->value.GetString());
  }

  if (const auto it = value.FindMember("points"); it != value.MemberEnd()) {
    anno.polygon = deserializeListOfPoints(it->value);
  }

  if (const auto it = value.FindMember("color"); it != value.MemberEnd()) {
    anno.color = QColor::fromString(it->value.GetString());
  }

  if (const auto it = value.FindMember("name"); it != value.MemberEnd()) {
    const auto s = QString::fromLocal8Bit(it->value.GetString());
    anno.name = s;
  }

  return anno;
}

} // namespace annotation
