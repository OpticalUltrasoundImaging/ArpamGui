#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

struct TimeIt {
  using clock = std::chrono::high_resolution_clock;
  TimeIt(const std::string &name) : name(name), start(clock::now()) {}
  ~TimeIt() {
    using namespace std::chrono;
    const auto elapsed = clock::now() - start;
    const auto nano = duration_cast<nanoseconds>(elapsed).count();
    std::cout << name << " " << (double)nano / 1.0e6 << " ms\n";
  }
  std::string name;
  clock::time_point start;
};

template <typename Func>
auto bench(const std::string &name, const int runs, const Func &func,
           bool write_to_file = false) {
  using namespace std::chrono;
  high_resolution_clock clock{};

  // Time it and collect data
  std::vector<int64_t> nanos(runs);
  for (int i = 0; i < runs; ++i) {
    const auto start = clock.now();
    func();
    nanos[i] = duration_cast<nanoseconds>(clock.now() - start).count();
  }

  // Compute statistics
  const auto getMean = [](const std::vector<int64_t> &data) {
    const double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
  };

  const auto calculateStdDev = [](const std::vector<int64_t> &data,
                                  double mean) {
    const double sq_sum = std::accumulate(
        data.begin(), data.end(), 0.0, [mean](double acc, int64_t val) {
          // Accumulate the squared differences
          return acc + (val - mean) * (val - mean);
        });
    // Square root of average squared deviation
    return std::sqrt(sq_sum / data.size());
  };

  const double mean = getMean(nanos);
  const double std = calculateStdDev(nanos, mean);

  // Select best unit to display based on mean
  std::string unit;
  double scale;
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

  // Print message
  std::cout << name << " " << std::fixed << std::setprecision(2) << mean / scale
            << " " << unit << " ± " << std / scale << " " << unit
            << " per loop (mean ± std. dev. of " << runs << " runs)\n";

  if (write_to_file) {
    std::ofstream fs(name, std::ios::binary);
    if (fs.is_open()) {
      fs.write((char *)nanos.data(), nanos.size() * sizeof(int64_t));
    }
  }

  return nanos;
}
