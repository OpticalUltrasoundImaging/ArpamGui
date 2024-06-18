#include "uspam/io.hpp"
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

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

std::string IOParams::serializeToString() const {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  serializeToDoc().Accept(writer);
  return buffer.GetString();
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

bool IOParams::deserialize(const std::string &jsonString) {
  rapidjson::Document doc;
  if (doc.Parse(jsonString.c_str()).HasParseError()) {
    return false;
  }
  return deserialize(doc);
}

bool IOParams::serializeToFile(const fs::path &path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open " << path << " for writing.\n";
    return false;
  }
  ofs << this->serializeToString();
  return true;
}

bool IOParams::deserializeFromFile(const fs::path &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << path << " for reading.\n";
    return false;
  }

  const std::string jsonString((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());

  return deserialize(jsonString);
}

} // namespace uspam::io