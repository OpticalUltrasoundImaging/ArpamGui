#pragma once

#include <QImage>
#include <QLayout>
#include <QString>
#include <QtWidgets>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

struct ImshowCanvasCursorState {
  bool leftButtonDown = false;
  bool middleButtonDown = false;
  bool rightButtonDown = false;
  QPointF currPos;
  QPointF startPos;

  auto currLine() const { return QLineF(startPos, currPos); }
};

struct ImshowCanvasAnnotations {
  std::vector<QLineF> lines;
};

struct ImshowCanvasScaleBarState {
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

  /**
   * @brief Update the state of the scalebar
   * @param pixmapSize size of the pixmap on which the scalebar will be drawn
   * @param pix2mm [mm] size of one pixel
   */
  void update(QSize pixmapSize, double pix2mm) {
    minorTickLines.clear();
    minorTicks.clear();
    majorTicks.clear();
    majorTickLabels.clear();

    const auto pw = pixmapSize.width();
    const auto ph = pixmapSize.height();

    // Draw divisions in increments of 1 mm
    const double divisionSize = 1 / pix2mm;

    const int numDivisions = static_cast<int>(ph / (2 * divisionSize)) - 1;
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
   * @brief Draw the scalebar onto a pixmap painter
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

  void drawScaleBar(QPainter *painter);

  // [mm] Get distance (in real coordinates) between 2 points.
  double computeDistance_mm(QPointF pt1, QPointF pt2);

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

  // State of scalebars
  ImshowCanvasScaleBarState m_scalebar;

  // State of the cursor for drawing annotations
  ImshowCanvasCursorState m_cursorState;

  ImshowCanvasAnnotations m_anno;
};