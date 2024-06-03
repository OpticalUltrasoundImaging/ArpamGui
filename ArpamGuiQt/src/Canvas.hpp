#pragma once

#include "CanvasAnnotations.hpp"
#include "CanvasCursorState.hpp"
#include "CanvasTicks.hpp"
#include "geometryUtils.hpp"
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QLayout>
#include <QRect>
#include <QString>
#include <QTransform>
#include <QWidget>
#include <QtWidgets>
#include <opencv2/opencv.hpp>
#include <qgraphicsitem.h>
#include <qgraphicsscene.h>
#include <qtmetamacros.h>
#include <vector>

class Canvas : public QGraphicsView {
  Q_OBJECT

  Q_PROPERTY(CursorMode cursorMode READ cursorMode WRITE setCursorMode)

public:
  enum class CursorMode {
    LineMeasure = 0,
    BoxZoom,
  };
  Q_ENUM(CursorMode);

  explicit Canvas(QWidget *parent = nullptr);

  auto name() const { return m_name; }
  void setName(QString name) { m_name = std::move(name); }

  auto cursorMode() const { return m_cursorMode; }

public slots:
  void imshow(const cv::Mat &cv_img, double pix2m);
  void imshow(const QImage &img, double pix2m);
  void imshow(const QPixmap &pixmap, double pix2m);

  void setCursorMode(CursorMode mode) { m_cursorMode = mode; }

  // Roll back the last cursor action
  void undo();

signals:
  void error(QString err);

  // Signal emitted on mouseMoveEvent. Position is converted to the pixmap
  // domain
  void mouseMoved(QPoint pos, double depth_mm);

  void annoLineDrawn();
  void annoBoxDrawn();
  void annoFanDrawn();

protected:
  // Override event specifically to handle gesture events
  bool event(QEvent *event) override;

  void paintEvent(QPaintEvent *event) override;

  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

  void keyPressEvent(QKeyEvent *event) override;

private:
  bool gestureEvent(QGestureEvent *event);
  void pinchTriggered(QPinchGesture *gesture);

  // Convert mouse position to frame coordinates using the current offset and
  // scale
  inline auto widgetPosToFramePos(QPoint pos) {
    return (pos - m_offset) / m_scaleFactor;
  }
  inline auto framePosToWidgetPos(QPoint pos) {
    return pos * m_scaleFactor + m_offset;
  }

  void drawTicks(QPainter *painter);

  // [mm] Get distance between 2 points in the original // pixmap.
  double computeDistance_mm(QPointF pt1, QPointF pt2) const;
  // [mm] Get distance between 2 points in the scaled // pixmap.
  double computeDistanceScaled_mm(QPointF pt1, QPointF pt2) const;

  // Update m_scale, m_offset, and m_pixmapScaled
  void updateScaleOffsetAndScaledPixmap();

private:
  QGraphicsScene *m_scene;
  QString m_name;

  double m_scaleFactor{1.0}; // factor for m_pixmap to maintain aspect ratio
  QPixmap m_Pixmap;          // Original pixmap.
  QPixmap m_pixmapScaled;    // Cache of scaled pixmap

  QGraphicsPixmapItem *m_PixmapItem{nullptr};

  double m_pix2m{}; // [m] Factor converting pixel (in m_pixmap) to meters

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
  CanvasTicks m_ticks;

  // State of the cursor for drawing annotations
  CanvasCursorState m_cursor;
  CursorMode m_cursorMode{CursorMode::BoxZoom};

  CanvasAnnotations m_anno;
};