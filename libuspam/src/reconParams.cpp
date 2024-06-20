#include "uspam/reconParams.hpp"
#include "uspam/json.hpp"
#include <rapidjson/document.h>

namespace uspam::recon {

rapidjson::Document ReconParams2::serializeToDoc() const {
  using json::serializeArray;

  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  doc.AddMember("filterFreqPA", serializeArray(filterFreqPA, allocator),
                allocator);

  doc.AddMember("filterGainPA", serializeArray(filterGainPA, allocator),
                allocator);

  doc.AddMember("filterFreqUS", serializeArray(filterFreqUS, allocator),
                allocator);

  doc.AddMember("filterGainUS", serializeArray(filterGainUS, allocator),
                allocator);

  doc.AddMember("noiseFloorPA", noiseFloorPA, allocator);
  doc.AddMember("noiseFloorUS", noiseFloorUS, allocator);
  doc.AddMember("desiredDynamicRangePA", desiredDynamicRangePA, allocator);
  doc.AddMember("desiredDynamicRangeUS", desiredDynamicRangeUS, allocator);
  doc.AddMember("alineRotationOffset", alineRotationOffset, allocator);

  return doc;
}

bool ReconParams2::serializeToFile(const fs::path &path) const {
  return json::toFile(path, serializeToDoc());
}

bool ReconParams2::deserialize(const rapidjson::Document &doc) {
  using json::deserializeArray;

  if (doc.HasMember("filterFreqPA")) {
    assert(doc["filterFreqPA"].IsArray());
    deserializeArray(doc["filterFreqPA"], filterFreqPA);
  }
  if (doc.HasMember("filterGainPA")) {
    assert(doc["filterGainPA"].IsArray());
    deserializeArray(doc["filterGainPA"], filterGainPA);
  }
  if (doc.HasMember("filterFreqUS")) {
    assert(doc["filterFreqUS"].IsArray());
    deserializeArray(doc["filterFreqUS"], filterFreqUS);
  }
  if (doc.HasMember("filterGainUS")) {
    assert(doc["filterGainUS"].IsArray());
    deserializeArray(doc["filterGainUS"], filterGainUS);
  }

  noiseFloorPA = doc["noiseFloorPA"].GetInt();
  noiseFloorUS = doc["noiseFloorUS"].GetInt();
  desiredDynamicRangePA = doc["desiredDynamicRangePA"].GetInt();
  desiredDynamicRangeUS = doc["desiredDynamicRangeUS"].GetInt();
  alineRotationOffset = doc["alineRotationOffset"].GetInt();
  return true;
}

bool ReconParams2::deserializeFromFile(const fs::path &path) {
  rapidjson::Document doc;
  return json::fromFile(path, doc) && deserialize(doc);
}

} // namespace uspam::recon