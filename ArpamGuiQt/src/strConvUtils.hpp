#pragma once

#include <QString>
#include <filesystem>

namespace fs = std::filesystem;

// Convert a QString to a fs::path
inline auto qString2Path(const QString &str) {
  const auto utf8array = str.toUtf8();
  return fs::path(utf8array.constData());
}

inline auto path2string(const fs::path &path) {
#if defined(_WIN32) || defined(_WIN64)
  return path.wstring();
#else
  return path.string();
#endif
}

inline auto path2QString(const fs::path &path) {
#if defined(_WIN32) || defined(_WIN64)
  return QString::fromStdWString(path.wstring());
#else
  return QString::fromStdString(path.string());
#endif
}
