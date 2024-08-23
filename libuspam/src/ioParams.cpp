#include "uspam/ioParams.hpp"
#include "uspam/json.hpp"
#include <rapidjson/document.h>

namespace uspam::io {

rapidjson::Document IOParams::serializeToDoc() const {
  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  doc.AddMember("alinesPerBscan", alinesPerBscan, allocator);
  doc.AddMember("samplesPerAscan", samplesPerAscan, allocator);

  doc.AddMember("PAstart", PA.start, allocator);
  doc.AddMember("PAdelay", PA.delay, allocator);
  doc.AddMember("PAsize", PA.size, allocator);

  doc.AddMember("USstart", US.start, allocator);
  doc.AddMember("USdelay", US.delay, allocator);
  doc.AddMember("USsize", US.size, allocator);

  doc.AddMember("byteOffset", this->byteOffset, allocator);
  return doc;
}

bool IOParams::deserialize(const rapidjson::Document &doc) {

  this->alinesPerBscan = doc["alinesPerBscan"].GetInt();
  this->samplesPerAscan = doc["samplesPerAscan"].GetInt();

  this->PA.start = doc["PAstart"].GetInt();
  this->PA.delay = doc["PAdelay"].GetInt();
  this->PA.size = doc["PAsize"].GetInt();

  this->US.start = doc["USstart"].GetInt();
  this->US.delay = doc["USdelay"].GetInt();
  this->US.size = doc["USsize"].GetInt();

  this->byteOffset = doc["byteOffset"].GetInt();
  return true;
}

bool IOParams::serializeToFile(const fs::path &path) const {
  return json::toFile(path, serializeToDoc());
}

bool IOParams::deserializeFromFile(const fs::path &path) {
  rapidjson::Document doc;
  return json::fromFile(path, doc) && deserialize(doc);
}

} // namespace uspam::io
