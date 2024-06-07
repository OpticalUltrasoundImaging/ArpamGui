#pragma once

#include "CanvasAnnotationItem.hpp"
#include "CanvasAnnotationModel.hpp"
#include "CanvasCursorState.hpp"
#include "CanvasOverlay.hpp"
#include "CanvasTicks.hpp"
#include "geometryUtils.hpp"
#include <QAbstractListModel>
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
#include <stdexcept>
#include <vector>

// Canvas displays and image in a QGraphicsView
// and supports drawing annotations.
//
// Annotations are stored in AnnotationModel (model),
// and the Canvas acts as its view and controller.
class Canvas : public QGraphicsView {
  Q_OBJECT

  Q_PROPERTY(CursorMode cursorMode READ cursorMode WRITE setCursorMode)

public:
  enum class CursorMode {
    Default = 0, // Let the parent (QGraphicsView) handle the mouse event
    Pan,
    MeasureLine,
    LabelRect,
  };
  Q_ENUM(CursorMode);

  explicit Canvas(QWidget *parent = nullptr);

  void setModel(AnnotationModel *model) {
    this->m_annotations = model;
    connect(model, &AnnotationModel::dataChanged, this, &Canvas::onDataChanged);
  }

  auto overlay() { return m_overlay; }

  [[nodiscard]] auto cursorMode() const { return m_cursorMode; }

  // Update the scale to fit the pixmap in the widget
  void updateMinScaleFactor();
  void scaleToSize();

public slots: // NOLINT
  void imshow(const cv::Mat &cv_img, double pix2m);
  void imshow(const QImage &img, double pix2m);
  void imshow(const QPixmap &pixmap, double pix2m);

  void setCursorMode(CursorMode mode) { m_cursorMode = mode; }

  void onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight,
                     const QVector<int> &roles) {
    for (int row = topLeft.row(); row <= bottomRight.row(); ++row) {
      updateAnnotationItem(row);
    }
  }

  void onRowsInserted(const QModelIndex &parent, int first, int last) {
    Q_UNUSED(parent);
    for (int row = first; row <= last; ++row) {
      addAnnotationItem(row);
    }
  }

  void onRowsRemoved(const QModelIndex &parent, int first, int last) {
    Q_UNUSED(parent);
    for (int row = first; row <= last; ++row) {
      removeAnnotationItem(row);
    }
  }

  // Roll back the last cursor action
  void undo();

  void resetZoomOnNextImshow() { m_resetZoomOnNextImshow = true; }

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

  void wheelEvent(QWheelEvent *event) override;

  void resizeEvent(QResizeEvent *event) override;

private:
  // Called whenever m_scaleFactor changes
  void updateTransform();

  // Pinch zoom handlers
  bool gestureEvent(QGestureEvent *event);
  void pinchTriggered(QPinchGesture *gesture);

  // Panning
  void panStartEvent(QMouseEvent *event);
  void panMoveEvent(QMouseEvent *event);

  // Annotation handlers
  // Add an existing annotation in the model at `row`
  // to the view
  void addAnnotationItem(int row) {
    const Annotation &annotation = m_annotations->getAnnotation(row);

    QGraphicsItem *item = [&]() -> QGraphicsItem * {
      switch (annotation.type()) {

      case Annotation::Line: {
        auto *item = new QGraphicsLineItem(QLineF(
            annotation.rect().topLeft(), annotation.rect().bottomRight()));
        item->setPen(QPen(annotation.color()));
        return item;
      }

      case Annotation::Rect: {
        auto *item = new QGraphicsRectItem(annotation.rect());
        item->setPen(QPen(annotation.color()));
        return item;
      }

      case Annotation::Polygon: {
        auto *item = new QGraphicsPolygonItem(annotation.polygon());
        item->setPen(QPen(annotation.color()));
        return item;
      }
      default:
        throw std::runtime_error("Shouldn't get here");
      }
    }();

    scene()->addItem(item);
    m_annotationItems.append(item);
  }

  // Update an existing annotation in the model at `row`
  // to the view
  void updateAnnotationItem(int row) {
    QGraphicsItem *item = m_annotationItems[row];
    const Annotation &annotation = m_annotations->getAnnotation(row);

    switch (annotation.type()) {
    case Annotation::Line:
      if (auto *lineItem = dynamic_cast<QGraphicsLineItem *>(item);
          lineItem != nullptr) {
        lineItem->setLine(QLineF(annotation.rect().topLeft(),
                                 annotation.rect().bottomRight()));
        lineItem->setPen(QPen(annotation.color()));
      }
      break;

    case Annotation::Rect:
      if (auto *rectItem = dynamic_cast<QGraphicsRectItem *>(item);
          rectItem != nullptr) {
        rectItem->setRect(annotation.rect());
        rectItem->setPen(QPen(annotation.color()));
      }
      break;

    case Annotation::Polygon:
      if (auto *polygonItem = dynamic_cast<QGraphicsPolygonItem *>(item);
          polygonItem != nullptr) {
        polygonItem->setPolygon(annotation.polygon());
        polygonItem->setPen(QPen(annotation.color()));
      }
      break;
    }
  }

  void removeAnnotationItem(int row) {
    QGraphicsItem *item = m_annotationItems.takeAt(row);
    scene()->removeItem(item);
    delete item;
  }

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
  QPointF m_transformCenter;
  bool m_resetZoomOnNextImshow{true};

  // Panning
  QPointF m_lastPanPoint;

  // Image
  QPixmap m_Pixmap; // Image pixmap
  QGraphicsPixmapItem *m_PixmapItem{nullptr};

  double m_pix2m{}; // [m] Factor converting pixel (in m_pixmap) to meters

  // State of ticks
  CanvasTicks m_ticks;

  // State of the cursor for drawing annotations
  CanvasCursorState m_cursor;
  CursorMode m_cursorMode{CursorMode::LabelRect};

  // The graphics item currently being drawn by the cursor
  AnnotationGraphicsItemBase *m_currItem{nullptr};
  QGraphicsSimpleTextItem *m_currLabelItem{nullptr};

  AnnotationModel *m_annotations{};
  QList<QGraphicsItem *> m_annotationItems;
};
