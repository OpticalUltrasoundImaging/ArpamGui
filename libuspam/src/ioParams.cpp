#include "uspam/ioParams.hpp"
#include "uspam/json.hpp"
#include <rapidjson/document.h>

namespace uspam::io {

rapidjson::Document IOParams::serializeToDoc() const {
  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  doc.AddMember("rfSizePA", this->rf_size_PA, allocator);
  doc.AddMember("rfSizeSpacer", this->rf_size_spacer, allocator);
  doc.AddMember("rfSizeUS", this->rf_size_US, allocator);

  doc.AddMember("offsetUS", this->offsetUS, allocator);
  doc.AddMember("offsetPA", this->offsetPA, allocator);

  doc.AddMember("byteOffset", this->byte_offset, allocator);
  return doc;
}

bool IOParams::deserialize(const rapidjson::Document &doc) {
  this->rf_size_PA = doc["rfSizePA"].GetInt();
  this->rf_size_spacer = doc["rfSizeSpacer"].GetInt();
  this->rf_size_US = doc["rfSizeUS"].GetInt();

  this->offsetPA = doc["offsetPA"].GetInt();
  this->offsetUS = doc["offsetUS"].GetInt();

  this->byte_offset = doc["byteOffset"].GetInt();
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
