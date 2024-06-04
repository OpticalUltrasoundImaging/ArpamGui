#pragma once

#include "CanvasAnnotationModel.hpp"
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
#include <qabstractitemmodel.h>
#include <qcontainerfwd.h>
#include <qgraphicsitem.h>
#include <qgraphicsscene.h>
#include <qgraphicssceneevent.h>
#include <qtmetamacros.h>
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
    LineMeasure = 0,
    BoxZoom,
  };
  Q_ENUM(CursorMode);

  explicit Canvas(QWidget *parent = nullptr);

  auto name() const { return m_name; }
  void setName(QString name) { m_name = std::move(name); }

  void setModel(AnnotationModel *model) {
    this->m_annotations = model;
    connect(model, &AnnotationModel::dataChanged, this, &Canvas::onDataChanged);
  }

  auto cursorMode() const { return m_cursorMode; }

  // Update the scale to fit the pixmap in the widget
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

private:
  void updateTransform();

  // Pinch zoom handlers
  bool gestureEvent(QGestureEvent *event);
  void pinchTriggered(QPinchGesture *gesture);

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

      case Annotation::Box: {
        auto *item = new QGraphicsRectItem(annotation.rect());
        item->setPen(QPen(annotation.color()));
        return item;
      }
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

    case Annotation::Box:
      if (auto *rectItem = dynamic_cast<QGraphicsRectItem *>(item);
          rectItem != nullptr) {
        rectItem->setRect(annotation.rect());
        rectItem->setPen(QPen(annotation.color()));
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
  double computeDistance_mm(QPointF pt1, QPointF pt2) const;

  QGraphicsScene *m_scene;
  QString m_name;

  double m_scaleFactor{1.0}; // factor for global transform
  double m_scaleFactorMin{0.5};

  QPixmap m_Pixmap; // Image pixmap
  QGraphicsPixmapItem *m_PixmapItem{nullptr};

  double m_pix2m{}; // [m] Factor converting pixel (in m_pixmap) to meters

  QTransform transformForward;  // From original to scaled pixmap space
  QTransform transformBackward; // from scaled to original pixmap space

  // State of ticks
  CanvasTicks m_ticks;

  // State of the cursor for drawing annotations
  CanvasCursorState m_cursor;
  CursorMode m_cursorMode{CursorMode::BoxZoom};
  QGraphicsLineItem *m_currLineItem{nullptr};

  CanvasAnnotations m_anno;

  AnnotationModel *m_annotations{};
  QList<QGraphicsItem *> m_annotationItems;
};