#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace uspam {
/**
Use RAII to time a code block.
Example:

{
  Timeit timeit("Func name");
  func();
}

*/
template <bool PrintStdOut = false> struct TimeIt {
  using clock = std::chrono::high_resolution_clock;
  explicit TimeIt(std::string name = {})
      : name(std::move(name)), start(clock::now()) {}
  TimeIt(const TimeIt &) = delete;
  TimeIt(TimeIt &&) = delete;
  TimeIt &operator=(const TimeIt &) = delete;
  TimeIt &operator=(TimeIt &&) = delete;
  [[nodiscard]] float get_ms() const {
    using namespace std::chrono; // NOLINT(*-namespace)
    const auto elapsed = clock::now() - start;
    const auto nano = duration_cast<nanoseconds>(elapsed).count();
    constexpr float fct_nano2mili = 1.0e-6;
    return static_cast<float>(nano) * fct_nano2mili;
  }
  ~TimeIt() {
    if constexpr (PrintStdOut) {
      std::cout << name << " " << get_ms() << " ms\n";
    }
  }
  std::string name;
  clock::time_point start;
};

/**
Simple benchmark function
Example:
auto nanos = bench("Func name", n_runs, [&]() { Func(); }, true);
*/
template <typename Func>
auto bench(const std::string &name, const int runs, const Func &func,
           bool write_to_file = false) {
  using namespace std::chrono; // NOLINT(*-namespace)
  using clock = high_resolution_clock;

  // Time it and collect data
  std::vector<int64_t> nanos(runs);
  for (int i = 0; i < runs; ++i) {
    const auto start = clock::now();
    func();
    nanos[i] = duration_cast<nanoseconds>(clock::now() - start).count();
  }

  // Compute statistics
  const auto getMean = [](const std::vector<int64_t> &data) {
    const double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / static_cast<double>(data.size());
  };

  const auto calculateStdDev = [](const std::vector<int64_t> &data,
                                  double mean) {
    const double sq_sum = std::accumulate(
        data.begin(), data.end(), 0.0, [mean](double acc, int64_t val) {
          // Accumulate the squared differences
          const auto v = static_cast<double>(val);
          return acc + (v - mean) * (v - mean);
        });
    // Square root of average squared deviation
    return std::sqrt(sq_sum / static_cast<double>(data.size()));
  };

  const double mean = getMean(nanos);
  const double std = calculateStdDev(nanos, mean);

  // Select best unit to display based on mean
  std::string unit;
  double scale{};
  // NOLINTBEGIN
  if (mean < 1000.0) { // nanoseconds
    unit = "ns";
    scale = 1.0;
  } else if (mean < 1e6) { // microseconds
    unit = "us";
    scale = 1e3;
  } else { // milliseconds
    unit = "ms";
    scale = 1e6;
  }
  // NOLINTEND

  // Print message
  std::cout << name << " " << std::fixed << std::setprecision(2) << mean / scale
            << " " << unit << " � " << std / scale << " " << unit
            << " per loop (mean � std. dev. of " << runs << " runs)\n";

  if (write_to_file) {
    std::ofstream fs(name, std::ios::binary);
    if (fs.is_open()) {
      // NOLINTNEXTLINE
      fs.write((char *)nanos.data(), nanos.size() * sizeof(int64_t));
    }
  }

  return nanos;
}
} // namespace uspam
