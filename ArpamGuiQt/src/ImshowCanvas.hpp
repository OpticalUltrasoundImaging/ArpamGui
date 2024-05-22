#pragma once

#include "geometryUtils.hpp"
#include <QImage>
#include <QLayout>
#include <QRect>
#include <QString>
#include <QWidget>
#include <QtWidgets>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

struct ImshowCanvasCursorState {
  bool leftButtonDown = false;
  bool middleButtonDown = false;
  bool rightButtonDown = false;
  QPointF currPos; // current cursor position in scaled pixmap coord
  QPointF startPos;

  QPointF currPosOrigal; // cursor position in original pixmap coord

  auto getLine() const { return QLineF(startPos, currPos); }
  auto getRect() const {
    qreal x = qMin(currPos.x(), startPos.x());
    qreal y = qMin(currPos.y(), startPos.y());

    qreal w = qAbs(currPos.x() - startPos.x());
    qreal h = qAbs(currPos.y() - startPos.y());
    return QRectF(x, y, w, h);
  }
};

struct ImshowCanvasAnnotations {
  static constexpr double whiskerLength = 10;

  std::vector<QLineF> lines;        // Line annotations in original pixmap space
  std::vector<QLineF> linesScaled;  // Lines in scaled pixmap space
  std::vector<QLineF> lineWhiskers; // Line whiskers in scaled pixmap space

  std::vector<QRectF> rects;
  std::vector<QRectF> rectsScaled;

  void clearLines() {
    lines.clear();
    linesScaled.clear();
    lineWhiskers.clear();
  }

  void clearRects() {
    rects.clear();
    rectsScaled.clear();
  }

  void clear() {
    clearLines();
    clearRects();
  }

  bool empty() const { return lines.empty() && rects.empty(); }

  static auto computeLineWhisker(QLineF line) {
    std::array<QLineF, 2> whiskers;
    const auto normalVecWhiskerHalf =
        geometry::calcNormalVec(line) * (whiskerLength / 2);
    whiskers[0] = QLineF(line.p1() - normalVecWhiskerHalf,
                         line.p1() + normalVecWhiskerHalf);
    whiskers[1] = QLineF(line.p2() - normalVecWhiskerHalf,
                         line.p2() + normalVecWhiskerHalf);
    return whiskers;
  }

  // Should be called on widget resize to precompute scaled annotations
  void rescale(double scale) {
    // Re scale lines and whiskers
    linesScaled.clear();
    lineWhiskers.clear();
    for (const auto &line : lines) {
      const QLineF lineScaled(line.p1() * scale, line.p2() * scale);
      linesScaled.push_back(lineScaled);
      // TODO compute whiskers

      auto whiskers = computeLineWhisker(lineScaled);
      lineWhiskers.push_back(whiskers[0]);
      lineWhiskers.push_back(whiskers[1]);
    }

    rectsScaled.clear();
    for (const auto &rect : rects) {
      const QRectF rectScaled(rect.x() * scale, rect.y() * scale,
                              rect.width() * scale, rect.height() * scale);
      rectsScaled.push_back(rectScaled);
    }
  }
};

struct ImshowCanvasTicks {
  static constexpr int minorTickSize = 10; // [px]
  static constexpr int majorTickSize = 15; // [px]
  static constexpr int margin = 5;         // [px]

  // Display major tick every n ticks
  static constexpr int majorTickInterval = 10;

  std::vector<int> minorTicks; // position in m_pixmapScaled
  std::vector<QLine> minorTickLines;

  std::vector<int> majorTicks;
  std::vector<QLine> majorTickLines;

  std::vector<int> majorTickLabels;

  void clear() {
    minorTicks.clear();
    minorTickLines.clear();

    majorTicks.clear();
    majorTickLines.clear();
    majorTickLabels.clear();
  }

  /**
   * @brief Update the state of the scalebar
   * @param pixmapSize size of the pixmap on which the scalebar will be drawn
   * @param pix2mm [mm] size of one pixel
   */
  void update(QSize pixmapSize, double pix2mm) {
    clear();

    const auto pw = pixmapSize.width();
    const auto ph = pixmapSize.height();

    // Draw divisions in increments of 1 mm
    const double divisionSize = 1 / pix2mm;

    const int numDivisions = static_cast<int>(ph / (2 * divisionSize));
    // numDivisions = numDivisions - (numDivisions % displayInterval) + 1;

    for (int i = -numDivisions; i < numDivisions; ++i) {
      const auto pos = ph / 2 + static_cast<int>(i * divisionSize);
      if (i % majorTickInterval == 0) {
        majorTicks.push_back(pos);
        majorTickLabels.push_back(abs(i));

        // Vertical scale bar
        majorTickLines.emplace_back(0, pos, minorTickSize, pos);
        // Horizontal scale bar
        majorTickLines.emplace_back(pos, 0, pos, minorTickSize);
      } else {
        minorTicks.push_back(pos);

        // Vertical scale bar
        minorTickLines.emplace_back(0, pos, minorTickSize, pos);
        // Horizontal scale bar
        minorTickLines.emplace_back(pos, 0, pos, minorTickSize);
      }
    }
  }

  /**
   * @brief Draw the ticks onto a pixmap painter
   * @param painter
   */
  void draw(QPainter *painter) {
    // Draw minor ticks
    painter->setPen(Qt::gray);
    painter->drawLines(minorTickLines.data(), minorTickLines.size());

    // Draw major ticks
    painter->setPen(Qt::white);
    for (int i = 0; i < majorTicks.size(); ++i) {
      // Vertical scale bar (horizontal ticks)
      const auto pos = majorTicks[i];
      painter->drawText(majorTickSize + margin, pos + margin,
                        QString::number(majorTickLabels[i]));
      // painter->drawText(pos - margin, majorTickSize + margin * 3,
    }
    painter->drawLines(majorTickLines.data(), majorTickLines.size());
  }
};

class ImshowCanvas : public QLabel {
  Q_OBJECT
public:
  enum class CursorType {
    LineMeasure = 0,
    BoxZoom,
  };

  explicit ImshowCanvas(QWidget *parent = nullptr);

  void setName(QString name) { m_name = name; }

public slots:
  void imshow(const cv::Mat &cv_img, double pix2m);
  void imshow(const QImage &img, double pix2m);
  void imshow(const QPixmap &pixmap, double pix2m);

signals:
  void error(QString err);

  // Signal emitted on mouseMoveEvent. Position is converted to the pixmap
  // domain
  void mouseMoved(QPoint pos, double depth_mm);

protected:
  void paintEvent(QPaintEvent *event) override;

  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

private:
  // Convert mouse position to frame coordinates using the current offset and
  // scale
  inline auto widgetPosToFramePos(QPoint pos) {
    return (pos - m_offset) / m_scale;
  }
  inline auto framePosToWidgetPos(QPoint pos) {
    return pos * m_scale + m_offset;
  }

  void drawTicks(QPainter *painter);

  // [mm] Get distance between 2 points in the original // pixmap.
  double computeDistance_mm(QPointF pt1, QPointF pt2) const;
  // [mm] Get distance between 2 points in the scaled // pixmap.
  double computeDistanceScaled_mm(QPointF pt1, QPointF pt2) const;

private:
  QString m_name;

  QPixmap m_pixmap;       // Original pixmap.
  QPixmap m_pixmapScaled; // Cache of scaled pixmap

  double m_pix2m{}; // [m] Factor converting pixel (in m_pixmap) to meters
  double m_scale{}; // factor for m_pixmap to maintain aspect ratio

  // Offset of displayed scaled m_pixmap to keep center.
  // m_offset is updated on every paintEvent so it should never be zero
  // when a m_pixmap is present.
  QPoint m_offset{};

  QRectF m_zoomRect;
  bool m_zoomed{false};
  bool m_zoomTranslated{false};

  // State of ticks
  ImshowCanvasTicks m_ticks;

  // State of the cursor for drawing annotations
  ImshowCanvasCursorState m_cursor;

  CursorType m_cursorType{CursorType::BoxZoom};

  ImshowCanvasAnnotations m_anno;
};