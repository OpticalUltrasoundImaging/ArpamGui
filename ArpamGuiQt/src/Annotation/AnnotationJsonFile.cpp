#include "Annotation/AnnotationJsonFile.hpp"
#include "Annotation/Annotation.hpp"
#include "datetime.hpp"
#include <QDebug>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/rapidjson.h>
#include <uspam/json.hpp>
#include <utility>

namespace annotation {

AnnotationJsonFile::AnnotationJsonFile() { init(); }

bool AnnotationJsonFile::readFromFile(const fs::path &path) {
  uspam::json::fromFile(path, m_doc);

  if (m_doc.HasParseError()) {
    qDebug() << "Parse Error at offset " << m_doc.GetErrorOffset() << ". "
             << rapidjson::GetParseError_En(m_doc.GetParseError());
    return false;
  }
  return true;
}

void AnnotationJsonFile::writeToFile(const fs::path &path) const {
  uspam::json::toFile(path, m_doc);
}

void AnnotationJsonFile::init() {
  // Create
  m_doc.SetObject();
  auto &allocator = m_doc.GetAllocator();

  {
    auto timeVal =
        uspam::json::serializeString(datetime::datetimeISO8601(), allocator);
    m_doc.AddMember("date-created", timeVal, allocator);
    m_doc.AddMember("date-modified", timeVal, allocator);
  }

  {
    rapidjson::Value docFrames(rapidjson::kObjectType);
    m_doc.AddMember("frames", docFrames, allocator);
  }
}

QList<Annotation> AnnotationJsonFile::getAnnotationForFrame(int frameNum) {

  assert(m_doc.HasMember("frames"));
  const auto &frames = m_doc["frames"];
  auto allocator = m_doc.GetAllocator();

  rapidjson::Value frameNumStr =
      uspam::json::serializeString(std::to_string(frameNum), allocator);

  QList<Annotation> result;
  if (frames.HasMember(frameNumStr)) {
    if (const auto &member = frames[frameNumStr]; !member.IsNull()) {
      assert(member.GetType() == rapidjson::kArrayType);

      const auto &jsonArray = member.GetArray();
      result.reserve(jsonArray.Size());

      for (const auto &value : jsonArray) {
        result.append(Annotation::deserializeFromJson(value));
      }
    }
  }
  return result;
}

void AnnotationJsonFile::setAnnotationForFrame(
    int frameNum, const QList<Annotation> &annotations) {

  auto &allocator = m_doc.GetAllocator();

  // Update date-modified
  m_doc["date-modified"] =
      uspam::json::serializeString(datetime::datetimeISO8601(), allocator);

  auto &frames = m_doc["frames"];

  rapidjson::Value frameNumStr =
      uspam::json::serializeString(std::to_string(frameNum), allocator);

  rapidjson::Value jsonArray(rapidjson::kArrayType);

  jsonArray.Reserve(annotations.size(), allocator);
  for (const auto &anno : annotations) {
    jsonArray.PushBack(anno.serializeToJson(allocator), allocator);
  }

  // Replace in the doc
  if (frames.HasMember(frameNumStr)) {
    frames.RemoveMember(frameNumStr);
  }
  frames.AddMember(frameNumStr, jsonArray, allocator);
}
} // namespace annotation