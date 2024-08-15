#pragma once

#include "Annotation/AnnotationModel.hpp"
#include "Annotation/GraphicsItemBase.hpp"
#include "CanvasCursorState.hpp"
#include "CanvasOverlay.hpp"
// #include "CanvasTicks.hpp"
#include <QAbstractListModel>
#include <QCursor>
#include <QEvent>
#include <QGraphicsItem>
#include <QGraphicsPolygonItem>
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include <QGraphicsSceneEvent>
#include <QGraphicsView>
#include <QGraphicsWidget>
#include <QImage>
#include <QLayout>
#include <QRect>
#include <QResizeEvent>
#include <QString>
#include <QTransform>
#include <QWheelEvent>
#include <QWidget>
#include <QtWidgets>
#include <opencv2/opencv.hpp>

// Canvas displays and image in a QGraphicsView
// and supports drawing annotations.
//
// Annotations are stored in AnnotationModel (model),
// and the Canvas acts as its view and controller.
class Canvas : public QGraphicsView {
  Q_OBJECT

  Q_PROPERTY(CursorMode cursorMode READ cursorMode WRITE setCursorMode)

public:
  using AnnotationModel = annotation::AnnotationModel;
  using Annotation = annotation::Annotation;

  enum class CursorMode {
    Default = 0, // Let QGraphicsView handle the mouse event
    Pan,
    SelectAScan,
    MeasureLine,
    LabelRect,
    LabelFan,
  };
  Q_ENUM(CursorMode);

  explicit Canvas(QWidget *parent = nullptr);

  void setModel(AnnotationModel *model);

  auto overlay() { return m_overlay; }

  [[nodiscard]] auto cursorMode() const { return m_cursorMode; }

  // Update the scale to fit the pixmap in the widget
  void updateMinScaleFactor();
  void scaleToSize();

public slots: // NOLINT

  /* Show image */
  void imshow(const cv::Mat &cv_img, double pix2m);
  void imshow(const QImage &img, double pix2m);
  void imshow(const QPixmap &pixmap, double pix2m);

  void setCursorMode(CursorMode mode);

  void onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight,
                     const QVector<int> &roles);
  void onRowsInserted(const QModelIndex &parent, int first, int last);
  void onRowsRemoved(const QModelIndex &parent, int first, int last);
  // void onRowsMoved(const QModelIndex &sourceParent, int sourceStart,
  //                  int sourceEnd, const QModelIndex &destinationParent,
  //                  int destinationRow); // Not implemented yet

  // Roll back the last cursor action
  void undo();

  void resetZoomOnNextImshow() { m_resetZoomOnNextImshow = true; }

  void setAlinesPerBscan(int alinesPerBscan) {
    m_alinesPerBscan = alinesPerBscan;
  }

signals:
  void error(QString err);

  // Signal emitted on mouseMoveEvent. Position is converted to the pixmap
  // domain
  void mouseMoved(QPoint pos, double depth_mm);

  // Signal emitted when mouse moved in selectAScan mode.
  void AScanSelected(int idx);

protected:
  // Override event specifically to handle gesture events
  bool event(QEvent *event) override;

  void paintEvent(QPaintEvent *event) override;

  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

  void wheelEvent(QWheelEvent *event) override;

  void resizeEvent(QResizeEvent *event) override;

private:
  // Must be called when m_scaleFactor changes
  void updateTransform();

  // Pinch zoom handlers
  bool gestureEvent(QGestureEvent *event);
  void pinchTriggered(QPinchGesture *gesture);

  // Panning handlers
  void panStartEvent(QMouseEvent *event);
  void panMoveEvent(QMouseEvent *event);
  void panEndEvent(QMouseEvent *event);

  // Annotation graphics handlers
  // Add an existing annotation in the model at `row`
  // to the view
  void addGraphicsItemFromModel(int row);

  // Update an existing annotation in the model at `row`
  // to the view
  void updateGraphicsItemFromModel(int row);

  void removeGraphicsItem(int row);

  // Clear m_currLabelItem and m_currItem
  void removeCurrItem();

  void drawTicks(QPainter *painter);

  // [mm] Get distance between 2 points in the original pixmap space.
  [[nodiscard]] double computeDistance_mm(QPointF pt1, QPointF pt2) const;

  // Scene
  QGraphicsScene *m_scene;

  // Scene overlay items
  CanvasOverlay *m_overlay;

  // Transform
  double m_scaleFactor{1.0}; // factor for global transform
  double m_scaleFactorMin{1.0};
  bool m_resetZoomOnNextImshow{true};

  // Panning
  QPointF m_lastPanPoint;
  QCursor m_panLastCursor;

  // Image
  QPixmap m_Pixmap; // Image pixmap
  QGraphicsPixmapItem *m_PixmapItem{nullptr};

  double m_pix2m{}; // [m] Factor converting pixel (in m_pixmap) to meters

  // State of the cursor for drawing annotations
  CanvasCursorState m_cursor;
  CursorMode m_cursorMode{CursorMode::SelectAScan};

  //
  int m_alinesPerBscan{1000};

  // The graphics item currently being drawn by the cursor
  annotation::GraphicsItemBase *m_currItem{nullptr};

  AnnotationModel *m_model{};
  QList<annotation::GraphicsItemBase *> m_graphicsItems;
};
