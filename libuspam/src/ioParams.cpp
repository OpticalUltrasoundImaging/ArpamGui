#include "uspam/ioParams.hpp"
#include "uspam/json.hpp"
#include <rapidjson/document.h>

namespace uspam::io {

rapidjson::Document IOParams::serializeToDoc() const {
  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  doc.AddMember("rfSizePA", this->rfSizePA, allocator);
  doc.AddMember("rfSizeSpacer", this->rfSizeSpacer, allocator);

  doc.AddMember("offsetUS", this->offsetUS, allocator);
  doc.AddMember("offsetPA", this->offsetPA, allocator);

  doc.AddMember("byteOffset", this->byteOffset, allocator);
  return doc;
}

bool IOParams::deserialize(const rapidjson::Document &doc) {
  this->rfSizePA = doc["rfSizePA"].GetInt();
  this->rfSizeSpacer = doc["rfSizeSpacer"].GetInt();

  this->offsetPA = doc["offsetPA"].GetInt();
  this->offsetUS = doc["offsetUS"].GetInt();

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
