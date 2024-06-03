#include "CanvasAnnotations.hpp"

void CanvasAnnotations::Lines::clear() noexcept {
  lines.clear();
  scaled.clear();
  whiskers.clear();
}

bool CanvasAnnotations::Lines::empty() const noexcept { return lines.empty(); }

auto CanvasAnnotations::Lines::computeLineWhisker(QLineF line)
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

void CanvasAnnotations::Lines::addScaled(QLineF lineScaled, double scale,
                                         QPointF offset) {
  const QLineF line(lineScaled.p1() / scale + offset,
                    lineScaled.p2() / scale + offset);
  scaled.push_back(lineScaled);
  lines.push_back(line);

  const auto _whiskers = computeLineWhisker(lineScaled);
  whiskers.push_back(_whiskers[0]);
  whiskers.push_back(_whiskers[1]);
}

void CanvasAnnotations::Lines::addScaled(QLineF lineScaled,
                                         QTransform transformBackward) {
  scaled.push_back(lineScaled);
  lines.push_back(transformBackward.map(lineScaled));

  const auto _whiskers = computeLineWhisker(lineScaled);
  whiskers.push_back(_whiskers[0]);
  whiskers.push_back(_whiskers[1]);
}

void CanvasAnnotations::Lines::pop() {
  lines.pop_back();
  scaled.pop_back();
  whiskers.pop_back();
  whiskers.pop_back();
}

void CanvasAnnotations::Lines::rescale(double scale, QPointF offset) {
  // Re scale lines and whiskers
  scaled.clear();
  whiskers.clear();
  for (const auto &line : lines) {
    const QLineF lineScaled((line.p1() - offset) * scale,
                            (line.p2() - offset) * scale);
    scaled.push_back(lineScaled);

    auto _whiskers = computeLineWhisker(lineScaled);
    whiskers.push_back(_whiskers[0]);
    whiskers.push_back(_whiskers[1]);
  }
}

void CanvasAnnotations::Lines::rescale(QTransform forwardTransform) {
  scaled.clear();
  whiskers.clear();
  for (const auto &line : lines) {
    const auto lineScaled = forwardTransform.map(line);
    scaled.push_back(lineScaled);

    auto _whiskers = computeLineWhisker(lineScaled);
    whiskers.push_back(_whiskers[0]);
    whiskers.push_back(_whiskers[1]);
  }
}

void CanvasAnnotations::Rects::clear() noexcept {
  rects.clear();
  scaled.clear();
}

bool CanvasAnnotations::Rects::empty() const noexcept { return rects.empty(); }

void CanvasAnnotations::Rects::pop() noexcept {
  rects.pop_back();
  scaled.pop_back();
}

void CanvasAnnotations::Rects::rescale(double scale, QPointF offset) {
  scaled.clear();
  for (const auto &rect : rects) {
    const QRectF rectScaled(rect.x() * scale, rect.y() * scale,
                            rect.width() * scale, rect.height() * scale);
    scaled.push_back(rectScaled);
  }
}

void CanvasAnnotations::Rects::rescale(QTransform forwardTransform) {
  scaled.clear();
  for (const auto &rect : rects) {
    scaled.push_back(forwardTransform.mapRect(rect));
  }
}

void CanvasAnnotations::clear() noexcept {
  lines.clear();
  rects.clear();
}

bool CanvasAnnotations::empty() const noexcept {
  return lines.empty() && rects.empty();
}

void CanvasAnnotations::rescale(double scale, QPointF offset) {
  lines.rescale(scale, offset);
  rects.rescale(scale, offset);
}

void CanvasAnnotations::rescale(QTransform forwardTransform) {
  lines.rescale(forwardTransform);
  rects.rescale(forwardTransform);
}