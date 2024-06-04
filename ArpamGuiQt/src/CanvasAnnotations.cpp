#include "CanvasAnnotations.hpp"

void CanvasAnnotations::Lines::clear() noexcept { lines.clear(); }

bool CanvasAnnotations::Lines::empty() const noexcept { return lines.empty(); }

void CanvasAnnotations::Lines::add(QLineF line) { lines.push_back(line); }

void CanvasAnnotations::Lines::pop() { lines.pop_back(); }

void CanvasAnnotations::Rects::clear() noexcept {
  rects.clear();
  scaled.clear();
}

bool CanvasAnnotations::Rects::empty() const noexcept { return rects.empty(); }

void CanvasAnnotations::Rects::pop() noexcept {
  rects.pop_back();
  scaled.pop_back();
}

void CanvasAnnotations::clear() noexcept {
  lines.clear();
  rects.clear();
}

bool CanvasAnnotations::empty() const noexcept {
  return lines.empty() && rects.empty();
}
