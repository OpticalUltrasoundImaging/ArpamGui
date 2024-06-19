#include "jsonUtils.hpp"
#include <array>
#include <cstdio>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/writer.h>

namespace jsonUtils {

void fromFile(const fs::path &path, Document &doc) {
  FILE *fp = std::fopen(path.c_str(), "r");

  std::array<char, 65536> buf;
  FileReadStream is(fp, buf.data(), buf.size());

  Document d;
  d.ParseStream(is);

  fclose(fp);
}

void toFile(const fs::path &path, const Document &doc) {
  FILE *fp = std::fopen(path.c_str(), "w");

  std::array<char, 65536> buf;
  FileWriteStream os(fp, buf.data(), buf.size());

  Writer<FileWriteStream> writer(os);
  doc.Accept(writer);

  fclose(fp);
}

} // namespace jsonUtils