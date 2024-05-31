#pragma once

#include "geometryUtils.hpp"
#include <QLineF>
#include <QPointF>
#include <QRectF>
#include <QTransform>
#include <array>
#include <vector>

struct ImshowCanvasAnnotations {
  struct Lines {
    std::vector<QLineF> lines;    // Line annotations in original pixmap space
    std::vector<QLineF> scaled;   // Lines in scaled pixmap space
    std::vector<QLineF> whiskers; // Line whiskers in scaled pixmap space

    void clear() noexcept;
    [[nodiscard]] bool empty() const noexcept;
    void pop();

    static auto computeLineWhisker(QLineF line) -> std::array<QLineF, 2>;

    void addScaled(QLineF lineScaled, double scale, QPointF offset);
    void addScaled(QLineF lineScaled, QTransform transformBackward);

    void rescale(double scale, QPointF offset);
    void rescale(QTransform forwardTransform);
  };

  struct Rects {
    std::vector<QRectF> rects;
    std::vector<QRectF> scaled;

    void clear() noexcept;
    [[nodiscard]] bool empty() const noexcept;
    void pop() noexcept;

    void rescale(double scale, QPointF offset);
    void rescale(QTransform forwardTransform);
  };

public:
  static constexpr double whiskerLength = 10;

  Lines lines;
  Rects rects;

public:
  void clear() noexcept;
  bool empty() const noexcept;

  // Should be called on widget resize to precompute scaled annotations
  void rescale(double scale, QPointF offset);
  void rescale(QTransform forwardTransform);
};