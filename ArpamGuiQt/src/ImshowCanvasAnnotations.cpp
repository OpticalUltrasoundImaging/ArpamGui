#include "ImshowCanvasAnnotations.hpp"

void ImshowCanvasAnnotations::Lines::clear() noexcept {
  lines.clear();
  scaled.clear();
  whiskers.clear();
}

bool ImshowCanvasAnnotations::Lines::empty() const noexcept {
  return lines.empty();
}

auto ImshowCanvasAnnotations::Lines::computeLineWhisker(QLineF line)
    -> std::array<QLineF, 2> {
  std::array<QLineF, 2> whiskers;
  const auto normalVecWhiskerHalf =
      geometry::calcNormalVec(line) * (whiskerLength / 2);
  whiskers[0] = QLineF(line.p1() - normalVecWhiskerHalf,
                       line.p1() + normalVecWhiskerHalf);
  whiskers[1] = QLineF(line.p2() - normalVecWhiskerHalf,
                       line.p2() + normalVecWhiskerHalf);
  return whiskers;
}

void ImshowCanvasAnnotations::Lines::addScaled(QLineF lineScaled,
                                               double scale) {
  const QLineF line(lineScaled.p1() / scale, lineScaled.p2() / scale);
  scaled.push_back(lineScaled);
  lines.push_back(line);

  const auto _whiskers = computeLineWhisker(lineScaled);
  whiskers.push_back(_whiskers[0]);
  whiskers.push_back(_whiskers[1]);
}

void ImshowCanvasAnnotations::Lines::pop() {
  lines.pop_back();
  scaled.pop_back();
  whiskers.pop_back();
  whiskers.pop_back();
}

void ImshowCanvasAnnotations::Lines::rescale(double scale) {
  // Re scale lines and whiskers
  scaled.clear();
  whiskers.clear();
  for (const auto &line : lines) {
    const QLineF lineScaled(line.p1() * scale, line.p2() * scale);
    scaled.push_back(lineScaled);

    auto _whiskers = computeLineWhisker(lineScaled);
    whiskers.push_back(_whiskers[0]);
    whiskers.push_back(_whiskers[1]);
  }
}

void ImshowCanvasAnnotations::Rects::clear() noexcept {
  rects.clear();
  scaled.clear();
}

bool ImshowCanvasAnnotations::Rects::empty() const noexcept {
  return rects.empty();
}

void ImshowCanvasAnnotations::Rects::pop() noexcept {
  rects.pop_back();
  scaled.pop_back();
}

void ImshowCanvasAnnotations::Rects::rescale(double scale) {
  scaled.clear();
  for (const auto &rect : rects) {
    const QRectF rectScaled(rect.x() * scale, rect.y() * scale,
                            rect.width() * scale, rect.height() * scale);
    scaled.push_back(rectScaled);
  }
}

void ImshowCanvasAnnotations::clear() noexcept {
  lines.clear();
  rects.clear();
}

bool ImshowCanvasAnnotations::empty() const noexcept {
  return lines.empty() && rects.empty();
}

void ImshowCanvasAnnotations::rescale(double scale) {
  lines.rescale(scale);
  rects.rescale(scale);
}