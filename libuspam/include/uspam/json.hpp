#pragma once
#include <concepts>
#include <filesystem>
#include <rapidjson/document.h>
#include <type_traits>
#include <vector>

namespace uspam::json {
using namespace rapidjson; // NOLINT
namespace fs = std::filesystem;

/* Serialization helpers */

[[nodiscard]] inline auto serializeString(const std::string &str,
                                          Document::AllocatorType &allocator) {
  Value value;
  value.SetString(str.c_str(), allocator);
  return value;
}

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

template <typename T>
concept SequenceOfArithmetic =
    SequentialContainer<T> && std::is_arithmetic_v<typename T::value_type>;

template <SequenceOfArithmetic T>
auto serializeArray(const T &vec,
                    rapidjson::Document::AllocatorType &allocator) {
  rapidjson::Value jsonArray(rapidjson::kArrayType);
  for (double val : vec) {
    jsonArray.PushBack(val, allocator);
  }
  return jsonArray;
}

template <SequenceOfArithmetic T>
void deserializeArray(const rapidjson::Value &jsonValue, T &vec) {
  using VT = T::value_type;

  vec.clear();
  if (jsonValue.IsArray()) {
    for (const auto &v : jsonValue.GetArray()) {
      VT x;
      if constexpr (std::is_floating_point_v<VT>) {
        x = v.GetDouble();
      } else if constexpr (std::is_integral_v<VT>) {
        x = v.GetInt();
      } else {
        static_assert(std::is_arithmetic_v<VT>,
                      "The value type must be arithmetic.");
      }
      vec.push_back(x);
    }
  }
}

/* Deserialization helpers */

/* File IO */

bool fromFile(const fs::path &path, Document &doc);
bool toFile(const fs::path &path, const Document &doc);

} // namespace uspam::json