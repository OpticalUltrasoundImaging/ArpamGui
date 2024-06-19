#include <QList>
#include <QPoint>
#include <QPointF>
#include <QString>
#include <__iterator/concepts.h>
#include <concepts>
#include <rapidjson/document.h>
#include <string>
#include <type_traits>

namespace jsonUtils {

/* Serialization helpers */

[[nodiscard]] inline auto
serializeString(const std::string &str,
                rapidjson::Document::AllocatorType &allocator) {
  rapidjson::Value value;
  value.SetString(str.c_str(), allocator);
  return value;
}

[[nodiscard]] inline auto
serializeString(const QString &str,
                rapidjson::Document::AllocatorType &allocator) {
  return serializeString(str.toStdString(), allocator);
}

template <typename T>
concept Point = std::is_same_v<T, QPoint> || std::is_same_v<T, QPointF>;

// Concept to check if T has begin() and end() methods
template <typename T>
concept HasBeginEnd = requires(T t) {
                        { t.begin() } -> std::input_or_output_iterator;
                        { t.end() } -> std::input_or_output_iterator;
                      };

// Concept to check if size_type is a value member type
template <typename T>
concept HasSizeType = requires { typename T::size_type; };

// Concept to check if T is an iterable sequential container
template <typename T>
concept SequentialContainer = HasBeginEnd<T> && HasSizeType<T>;

// Concept to check if T is a sequential container of points
template <typename T>
concept SequentialContainerOfPoints =
    SequentialContainer<T> && Point<typename T::value_type>;

template <SequentialContainerOfPoints Container>
[[nodiscard]] auto
serializeListOfPoints(const Container &points,
                      rapidjson::Document::AllocatorType &allocator) {
  rapidjson::Value jsonArray(rapidjson::kArrayType);
  for (const auto &point : points) {
    rapidjson::Value p(rapidjson::kArrayType);
    p.PushBack(point.x(), allocator);
    p.PushBack(point.y(), allocator);
    jsonArray.PushBack(p, allocator);
  }
  return jsonArray;
}

/* Deserialization helpers */

} // namespace jsonUtils