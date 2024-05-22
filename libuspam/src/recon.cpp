#include "uspam/recon.hpp"
#include "uspam/signal.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace {
void serializeVector(rapidjson::Document::AllocatorType &allocator,
                     rapidjson::Value &jsonValue,
                     const std::vector<double> &vec) {
  for (double val : vec) {
    jsonValue.PushBack(val, allocator);
  }
}

void deserializeVector(const rapidjson::Value &jsonValue,
                       std::vector<double> &vec) {
  vec.clear();
  for (const auto &v : jsonValue.GetArray()) {
    vec.push_back(v.GetDouble());
  }
}
} // namespace

namespace uspam::recon {

void recon(const arma::mat &rf, const arma::vec &kernel, arma::mat &env) {
  const cv::Range range(0, static_cast<int>(rf.n_cols));
  // cv::parallel_for_(cv::Range(0, rf.n_cols), [&](const cv::Range &range) {
  arma::vec rf_filt(rf.n_rows);
  for (int i = range.start; i < range.end; ++i) {
    const auto src = rf.unsafe_col(i);
    auto dst = env.unsafe_col(i);
    fftconv::oaconvolve_fftw_same<double>(src, kernel, rf_filt);
    signal::hilbert_abs_r2c(rf_filt, dst);
  }
  // });
}

// Serialize to JSON
std::string ReconParams2::serialize() const {
  const auto &params = *this;
  rapidjson::Document doc;
  doc.SetObject();
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  rapidjson::Value filterFreqPA(rapidjson::kArrayType);
  serializeVector(allocator, filterFreqPA, params.filterFreqPA);
  doc.AddMember("filterFreqPA", filterFreqPA, allocator);

  rapidjson::Value filterGainPA(rapidjson::kArrayType);
  serializeVector(allocator, filterGainPA, params.filterGainPA);
  doc.AddMember("filterGainPA", filterGainPA, allocator);

  rapidjson::Value filterFreqUS(rapidjson::kArrayType);
  serializeVector(allocator, filterFreqUS, params.filterFreqUS);
  doc.AddMember("filterFreqUS", filterFreqUS, allocator);

  rapidjson::Value filterGainUS(rapidjson::kArrayType);
  serializeVector(allocator, filterGainUS, params.filterGainUS);
  doc.AddMember("filterGainUS", filterGainUS, allocator);

  doc.AddMember("noiseFloorPA", params.noiseFloorPA, allocator);
  doc.AddMember("noiseFloorUS", params.noiseFloorUS, allocator);
  doc.AddMember("desiredDynamicRangePA", params.desiredDynamicRangePA,
                allocator);
  doc.AddMember("desiredDynamicRangeUS", params.desiredDynamicRangeUS,
                allocator);
  doc.AddMember("alineRotationOffset", params.alineRotationOffset, allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  return buffer.GetString();
}

// Deserialize from JSON
bool ReconParams2::deserialize(const std::string &jsonString) {
  auto &params = *this;
  rapidjson::Document doc;
  if (doc.Parse(jsonString.c_str()).HasParseError()) {
    return false;
  }

  deserializeVector(doc["filterFreqPA"], params.filterFreqPA);
  deserializeVector(doc["filterGainPA"], params.filterGainPA);
  deserializeVector(doc["filterFreqUS"], params.filterFreqUS);
  deserializeVector(doc["filterGainUS"], params.filterGainUS);

  params.noiseFloorPA = doc["noiseFloorPA"].GetInt();
  params.noiseFloorUS = doc["noiseFloorUS"].GetInt();
  params.desiredDynamicRangePA = doc["desiredDynamicRangePA"].GetInt();
  params.desiredDynamicRangeUS = doc["desiredDynamicRangeUS"].GetInt();
  params.alineRotationOffset = doc["alineRotationOffset"].GetInt();

  return true;
}

bool ReconParams2::serializeToFile(const fs::path &path) const {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open " << path << " for writing.\n";
    return false;
  }

  ofs << this->serialize();
  return true;
}

bool ReconParams2::deserializeFromFile(const fs::path &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << path << " for reading.\n";
    return false;
  }

  const std::string jsonString((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());

  return deserialize(jsonString);
}

void ReconParams2::reconOneScan(io::PAUSpair<double> &rf,
                                io::PAUSpair<uint8_t> &rfLog, bool flip) const {

  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf.PA);
    imutil::fliplr_inplace(rf.US);

    // Do rotate
    const auto rotate_offset = this->alineRotationOffset;
    rf.PA = arma::shift(rf.PA, rotate_offset, 1);
    rf.US = arma::shift(rf.US, rotate_offset, 1);
  }

  // compute filter kernels
  const auto kernelPA = signal::firwin2(95, filterFreqPA, filterGainPA);
  const auto kernelUS = signal::firwin2(95, filterFreqUS, filterGainUS);

  auto env = io::PAUSpair<double>::empty_like(rf);

  recon(rf.PA, kernelPA, env.PA);
  logCompress<double>(env.PA, rfLog.PA, this->noiseFloorPA,
                      this->desiredDynamicRangePA);

  recon(rf.US, kernelUS, env.US);
  logCompress<double>(env.US, rfLog.US, this->noiseFloorUS,
                      this->desiredDynamicRangeUS);
}

auto ReconParams2::reconOneScan(io::PAUSpair<double> &rf,
                                bool flip) const -> io::PAUSpair<uint8_t> {
  auto rfLog = io::PAUSpair<uint8_t>::zeros_like(rf);
  reconOneScan(rf, rfLog, flip);
  return rfLog;
}

void ReconParams::reconOneScan(arma::Mat<double> &rf, arma::Mat<uint8_t> &rfLog,
                               bool flip) const {
  if (flip) {
    // Do flip
    imutil::fliplr_inplace(rf);

    // Do rotate
    rf = arma::shift(rf, rotateOffset, 1);
  }

  // compute filter kernels
  const auto kernel = signal::firwin2(95, filterFreq, filterGain);

  arma::Mat<double> env(rf.n_rows, rf.n_cols, arma::fill::none);

  recon(rf, kernel, env);
  logCompress<double>(env, rfLog, noiseFloor, desiredDynamicRange);
}

} // namespace uspam::recon
