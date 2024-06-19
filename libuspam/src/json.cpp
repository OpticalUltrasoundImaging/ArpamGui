#include "uspam/json.hpp"
#include <array>
#include <cstdio>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/writer.h>

namespace uspam::json {

// NOLINTBEGIN
bool fromFile(const fs::path &path, Document &doc) {
  FILE *fp = std::fopen(path.c_str(), "r");
  if (fp == nullptr) {
    return false;
  }

  std::array<char, 65536> buf;
  FileReadStream is(fp, buf.data(), buf.size());

  doc.ParseStream(is);

  fclose(fp);
  return true;
}

bool toFile(const fs::path &path, const Document &doc) {
  FILE *fp = std::fopen(path.c_str(), "w");
  if (fp == nullptr) {
    return false;
  }

  std::array<char, 65536> buf;
  FileWriteStream os(fp, buf.data(), buf.size());

  Writer<FileWriteStream> writer(os);
  doc.Accept(writer);

  fclose(fp);
  return true;
}
// NOLINTEND

} // namespace uspam::json