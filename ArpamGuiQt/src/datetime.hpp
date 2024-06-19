#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace datetime {

inline auto datetime() {
  const auto now = std::chrono::system_clock::now();
  const auto now_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32) || defined(_WIN64)
  localtime_s(&tm, &now_time_t); // Use localtime_s on Windows
#else
  localtime_r(&now_time_t, &tm); // Use localtime_r on other platforms
#endif
  return tm;
}

inline auto datetimeFormat(const char *format) {
  const auto now_tm = datetime();
  std::ostringstream oss;
  oss << std::put_time(&now_tm, format);
  return oss.str();
}

inline auto dateISO8601() { return datetimeFormat("%Y-%m-%d"); }
inline auto datetimeISO8601() { return datetimeFormat("%Y-%m-%dT%H:%M:%S"); }

} // namespace datetime