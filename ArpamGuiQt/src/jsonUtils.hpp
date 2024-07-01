#pragma once

#include <QList>
#include <QPoint>
#include <QPointF>
#include <QString>
#include <rapidjson/document.h>
#include <type_traits>
#include <uspam/json.hpp>

namespace jsonUtils {
// NOLINTNEXTLINE(*-using-namespace)
using namespace rapidjson;
namespace fs = std::filesystem;

/* Serialization helpers */

[[nodiscard]] inline auto serializeString(const QString &str,
                                          Document::AllocatorType &allocator) {
  return uspam::json::serializeString(str.toStdString(), allocator);
}

template <typename T>
concept Point = std::is_same_v<T, QPoint> || std::is_same_v<T, QPointF>;

// Concept to check if T is a sequential container of points
template <typename T>
concept SequentialContainerOfPoints =
    uspam::json::SequentialContainer<T> && Point<typename T::value_type>;

template <SequentialContainerOfPoints Container>
[[nodiscard]] auto serializeListOfPoints(const Container &points,
                                         Document::AllocatorType &allocator) {
  Value jsonArray(kArrayType);
  for (const auto &point : points) {
    Value p(kArrayType);
    p.PushBack(point.x(), allocator);
    p.PushBack(point.y(), allocator);
    jsonArray.PushBack(p, allocator);
  }
  return jsonArray;
}

/* Deserialization helpers */

} // namespace jsonUtils
