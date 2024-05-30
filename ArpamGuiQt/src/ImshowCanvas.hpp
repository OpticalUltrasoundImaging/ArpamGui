#pragma once

#include "ImshowCanvasAnnotations.hpp"
#include "ImshowCanvasCursorState.hpp"
#include "ImshowCanvasTicks.hpp"
#include "geometryUtils.hpp"
#include <QImage>
#include <QLayout>
#include <QRect>
#include <QString>
#include <QTransform>
#include <QWidget>
#include <QtWidgets>
#include <opencv2/opencv.hpp>
#include <vector>

class ImshowCanvas : public QLabel {
  Q_OBJECT
public:
  enum class CursorMode {
    LineMeasure = 0,
    BoxZoom,
  };

  explicit ImshowCanvas(QWidget *parent = nullptr);

  auto name() const { return m_name; }
  void setName(QString name) { m_name = name; }

  auto cursorMode() const { return m_cursorMode; }
  void setCursorMode(CursorMode mode) { m_cursorMode = mode; }

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

  void keyPressEvent(QKeyEvent *event) override;

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

  bool m_zoomed{false};
  bool m_zoomTranslated{false};
  QRectF m_zoomRect;
  std::vector<QRectF> m_zoomRectHistory;

  QTransform transformForward;  // From original to scaled pixmap space
  QTransform transformBackward; // from scaled to original pixmap space

  // State of ticks
  ImshowCanvasTicks m_ticks;

  // State of the cursor for drawing annotations
  ImshowCanvasCursorState m_cursor;

  CursorMode m_cursorMode{CursorMode::BoxZoom};

  ImshowCanvasAnnotations m_anno;
};