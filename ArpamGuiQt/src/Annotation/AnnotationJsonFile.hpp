#pragma once

#include "Annotation/Annotation.hpp"
#include <QList>
#include <QObject>
#include <filesystem>
#include <rapidjson/document.h>
#include <uspam/json.hpp>

namespace annotation {
namespace fs = std::filesystem;

/**
Represents the json annotation file for a binfile
 */
class AnnotationJsonFile : public QObject {
  Q_OBJECT

public:
  AnnotationJsonFile();

  void writeToFile(const fs::path &path) const;
  bool readFromFile(const fs::path &path);

  void init();

  [[nodiscard]] QList<Annotation> getAnnotationForFrame(int frameNum);
  void setAnnotationForFrame(int frameNum,
                             const QList<Annotation> &annotations);

private:
  rapidjson::Document m_doc;
};

} // namespace annotation