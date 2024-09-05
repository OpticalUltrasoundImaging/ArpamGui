#include "uspam/reconParams.hpp"
#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/json.hpp"
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>

namespace uspam::recon {

rapidjson::Value
ReconParams::serialize(rapidjson::Document::AllocatorType &allocator) const {
  using json::serializeArray;

  rapidjson::Value obj(rapidjson::kObjectType);

  obj.AddMember("medfiltKsize", medfiltKsize, allocator);

  obj.AddMember("filterType", static_cast<int>(filterType), allocator);
  obj.AddMember("firTaps", firTaps, allocator);
  obj.AddMember("iirOrder", iirOrder, allocator);

  obj.AddMember("bpHighFreq", bpHighFreq, allocator);
  obj.AddMember("bpLowFreq", bpLowFreq, allocator);

  obj.AddMember("padding", padding, allocator);
  obj.AddMember("truncate", truncate, allocator);

  obj.AddMember("rotateOffset", rotateOffset, allocator);

  obj.AddMember("noiseFloor", noiseFloor_mV, allocator);
  obj.AddMember("desiredDynamicRange", desiredDynamicRange, allocator);

  obj.AddMember("flipOnEven", flipOnEven, allocator);

  obj.AddMember("showSurface", showSurface, allocator);
  obj.AddMember("cleanSurface", cleanSurface, allocator);
  obj.AddMember("additionalSamplesToCleanSurface",
                additionalSamplesToCleanSurface, allocator);

  {
    rapidjson::Value v;
    const auto sv = beamformer::BeamformerTypeToString(beamformerType);
    v.SetString(sv.data(), static_cast<rapidjson::SizeType>(sv.size()),
                allocator);
    obj.AddMember("beamformerType", v, allocator);
  }

  // TODO beamformer params

  return obj;
}

ReconParams ReconParams::deserialize(const rapidjson::Value &obj) {
  using json::deserializeArray;

  ReconParams params;

  params.medfiltKsize = obj["medfiltKsize"].GetInt();

  params.filterType = static_cast<FilterType>(obj["filterType"].GetInt());
  params.firTaps = obj["firTaps"].GetInt();
  params.iirOrder = obj["iirOrder"].GetInt();

  params.bpHighFreq = obj["bpHighFreq"].GetFloat();
  params.bpLowFreq = obj["bpLowFreq"].GetFloat();

  params.padding = obj["padding"].GetInt();
  params.truncate = obj["truncate"].GetInt();
  params.rotateOffset = obj["rotateOffset"].GetInt();

  params.noiseFloor_mV = obj["noiseFloor"].GetFloat();
  params.desiredDynamicRange = obj["desiredDynamicRange"].GetFloat();

  params.flipOnEven = obj["flipOnEven"].GetBool();
  params.showSurface = obj["showSurface"].GetBool();

  params.cleanSurface = obj["cleanSurface"].GetBool();
  params.additionalSamplesToCleanSurface =
      obj["additionalSamplesToCleanSurface"].GetInt();

  params.beamformerType =
      beamformer::BeamformerTypeFromString(obj["beamformerType"].GetString());

  return params;
}

rapidjson::Document ReconParams2::serializeToDoc() const {
  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  doc.AddMember("PA", PA.serialize(allocator), allocator);
  doc.AddMember("US", US.serialize(allocator), allocator);

  return doc;
}

bool ReconParams2::serializeToFile(const fs::path &path) const {
  return json::toFile(path, serializeToDoc());
}

bool ReconParams2::deserialize(const rapidjson::Document &doc) {
  auto &params = *this;

  if (const auto it = doc.FindMember("PA"); it != doc.MemberEnd()) {
    params.PA = ReconParams::deserialize(it->value);
  }

  if (const auto it = doc.FindMember("US"); it != doc.MemberEnd()) {
    params.US = ReconParams::deserialize(it->value);
  }

  return true;
}

bool ReconParams2::deserializeFromFile(const fs::path &path) {
  rapidjson::Document doc;
  return json::fromFile(path, doc) && deserialize(doc);
}

} // namespace uspam::recon