#pragma once
#include <array>
#include <exception>
#include <iterator>

namespace uspam {

template <typename T, int N = 2> class RingBuffer {
public:
  void push_back(const T &val) { data[getEnd()] = val; }

  auto pop_front() {
    if (size > 0) {
      --size;
      return data[begin++];
    }
    throw std::runtime_error("RingBuffer Empty");
  }

  auto &front() { return data[begin]; }
  auto const &front() const { return data[begin]; }
  auto &back() { return data[getEnd()]; }
  auto const &back() const { return data[getEnd()]; }

private:
  inline int getEnd() { return (begin + size) % data.size(); }

  std::array<T, N> data;
  int begin;
  int size;
};

} // namespace uspam
