#include "Canvas.hpp"
#include "CanvasAnnotationItem.hpp"
#include "geometryUtils.hpp"
#include <QGestureEvent>
#include <QGraphicsView>
#include <QGraphicsWidget>
#include <QHBoxLayout>
#include <QImage>
#include <QPainter>
#include <QPinchGesture>
#include <QVBoxLayout>
#include <QtCore>
#include <QtDebug>
#include <QtLogging>
#include <qgraphicsview.h>
#include <qlogging.h>
#include <qnamespace.h>
#include <uspam/timeit.hpp>

Canvas::Canvas(QWidget *parent)
    : QGraphicsView(parent), m_scene(new QGraphicsScene),
      m_overlay(new CanvasOverlay(viewport()))

{
  setBackgroundBrush(QBrush(Qt::black));

  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setAlignment(Qt::AlignCenter);

  // Hide scrollbars
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  // Enable mouse tracking
  setMouseTracking(true);
  setCursor(Qt::CrossCursor);
  setFocusPolicy(Qt::StrongFocus);

  // Enable pinch gesture handling
  // setDragMode(DragMode::ScrollHandDrag);
  viewport()->setAttribute(Qt::WA_AcceptTouchEvents);
  grabGesture(Qt::PinchGesture);

  // Graphics rendering parameters
  setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
  setScene(m_scene);

  // Hide the overlay at init since no images are shown yet.
  m_overlay->hide();
}

void Canvas::imshow(const cv::Mat &cv_img, double pix2m) {
  QImage qi(cv_img.data, cv_img.cols, cv_img.rows,
            static_cast<int>(cv_img.step), QImage::Format_BGR888);
  this->imshow(QPixmap::fromImage(qi), pix2m);
}

void Canvas::updateMinScaleFactor() {
  if (m_Pixmap.isNull()) [[unlikely]] {
    return;
  }

  const auto w = width();
  const auto h = height();
  const auto pw = m_Pixmap.width();
  const auto ph = m_Pixmap.height();

  // Calculate the scaleFactor needed to fill the viewport with the image
  // This is also the minimum scale factor
  const qreal scale =
      qMin(w / static_cast<qreal>(pw), h / static_cast<qreal>(ph));
  m_scaleFactorMin = scale;
}

void Canvas::scaleToSize() {
  updateMinScaleFactor();
  m_scaleFactor = m_scaleFactorMin;

  updateTransform();
}

void Canvas::imshow(const QImage &img, double pix2m) {
  this->imshow(QPixmap::fromImage(img), pix2m);
}

void Canvas::imshow(const QPixmap &pixmap, double pix2m) {
  m_Pixmap = pixmap;
  m_pix2m = pix2m;

  // Remove current image
  if (m_PixmapItem != nullptr) {
    m_scene->removeItem(m_PixmapItem);
    delete m_PixmapItem;
  }

  // Add image pixmap to scene
  m_PixmapItem = m_scene->addPixmap(m_Pixmap);

  if (m_resetZoomOnNextImshow) {
    scaleToSize();
    m_resetZoomOnNextImshow = false;
  }

  m_overlay->setSize(m_Pixmap.size());
  m_overlay->show();
}

// NOLINTBEGIN(*-casting, *-narrowing-conversions)

// Compute the distance between two points in the scaled pixmap domain
double Canvas::computeDistance_mm(QPointF pt1, QPointF pt2) const {
  const auto distance = geometry::calcMagnitude(pt1 - pt2);
  const auto distance_mm = distance * m_pix2m * 1000;
  return distance_mm;
}

bool Canvas::event(QEvent *event) {
  if (event->type() == QEvent::Gesture) {
    return gestureEvent(
        static_cast<QGestureEvent *>(event)); // NOLINT(*downcast)
  }
  return QGraphicsView::event(event);
}

void Canvas::paintEvent(QPaintEvent *event) {
  uspam::TimeIt timeit;

  m_overlay->move(0, 0);
  QGraphicsView::paintEvent(event);

  // QPainter painter(this);

  // // Draw canvas name
  // if (!m_name.isNull()) {
  //   const int margin = 10;

  //   QRect boundingRect = QRect(QPoint{}, m_pixmapScaled.size());
  //   boundingRect.adjust(0, 0, -margin, -margin);

  //   painter.setPen(Qt::white);
  //   painter.drawText(boundingRect, Qt::AlignRight | Qt::AlignBottom, m_name);
  // }

  // // Draw existing annotations
  // {
  //   painter.setPen(Qt::white);

  //   // Draw lines
  //   {
  //     painter.drawLines(m_anno.lines.scaled.data(),
  //     m_anno.lines.scaled.size());

  //     for (const auto &line : m_anno.lines.scaled) {
  //       const auto distance = computeDistanceScaled_mm(line.p1(), line.p2());
  //       const auto msg = QString("%1 mm").arg(distance);
  //       const auto textPos = line.p2() + QPointF(5, 5);
  //       painter.drawText(textPos, msg);
  //     }

  //     painter.drawLines(m_anno.lines.whiskers.data(),
  //                       m_anno.lines.whiskers.size());
  //   }

  //   // Draw rects
  //   {
  //     painter.drawRects(m_anno.rects.scaled.data(),
  //     m_anno.rects.scaled.size());
  //   }
  // }

  // // Draw curr annotation
  // switch (m_cursorMode) {
  // case CursorMode::LineMeasure: {

  //   if (m_cursor.leftButtonDown) {
  //     painter.setPen(Qt::white);
  //     const auto line = m_cursor.getLine();
  //     painter.drawLine(line);

  //     const auto distance = computeDistanceScaled_mm(line.p1(), line.p2());
  //     const auto msg = QString("%1 mm").arg(distance);
  //     const auto textPos = line.p2() + QPointF(5, 5);
  //     painter.drawText(textPos, msg);

  //     const auto whiskers = m_anno.lines.computeLineWhisker(line);
  //     painter.drawLines(whiskers.data(), whiskers.size());
  //   }
  //   break;
  // }

  // case CursorMode::BoxZoom: {
  //   if (m_cursor.leftButtonDown) {
  //     painter.setPen(Qt::white);

  //     const auto rect = m_cursor.getRect();
  //     painter.drawRect(rect);
  //   }

  //   break;
  // }
  // }

  // Measure rendering time
  {
    // const auto renderTime_ms = timeit.get_ms();
    // const auto msg = QString("Rendering time %1 ms").arg(renderTime_ms);
    // emit error(msg);
  }
}

void Canvas::undo() {
  // switch (m_cursorMode) {

  // case CursorMode::MeasureLine:
  //   break;

  // case CursorMode::LabelRect:
  //   break;

  // default:
  //   break;
  // }
}

void Canvas::mousePressEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  m_cursor.startPos = m_PixmapItem->mapFromScene(mapToScene(event->pos()));
  m_cursor.pos = m_cursor.startPos;

  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = true;

    switch (m_cursorMode) {
    case (CursorMode::Default):
      QGraphicsView::mousePressEvent(event);
      break;
    case CursorMode::Pan:
      panStartEvent(event);
      break;
    case (CursorMode::MeasureLine): {
      event->accept();

      removeCurrItem();

      const auto line = m_cursor.line();
      const auto color = Qt::white;

      {
        m_currItem = new LineItem(line, color);
        m_scene->addItem(m_currItem);
      }

      // Make simple text label item
      {
        QFont font;
        font.setPointSizeF(16.0 / m_scaleFactor);
        m_currLabelItem = m_scene->addSimpleText("", font);

        // QPen pen(Qt::white);
        // pen.setWidth(0);
        // m_currLabelItem->setPen(pen);
        m_currLabelItem->setBrush(QBrush(Qt::white));
      }
    } break;

    case CursorMode::LabelRect: {
      event->accept();

      removeCurrItem();

      const auto rect = m_cursor.rect();
      {
        m_currItem = new RectItem(rect, Qt::white);
        m_scene->addItem(m_currItem);
      }

    } break;
    case CursorMode::LabelFan: {
      event->accept();

      removeCurrItem();

      // Convert mouse pos to angle
      const auto angle = m_cursor.angle(m_Pixmap.rect());
      {
        m_currItem = new FanItem(m_Pixmap.rect(), {angle, 0}, Qt::white);
        m_scene->addItem(m_currItem);
      }

    } break;
    }

  } else if (event->button() == Qt::MiddleButton) {
    // Middle button pan
    m_cursor.middleButtonDown = true;
    panStartEvent(event);

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = true;
    // event->accept();
    // undo();

    QGraphicsView::mousePressEvent(event);
  }
}

void Canvas::mouseMoveEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  m_cursor.pos = m_PixmapItem->mapFromScene(mapToScene(event->pos()));

  // [px] Compute distance to center
  const auto center = m_Pixmap.rect().center();
  const auto distanceToCenter_mm = computeDistance_mm(center, m_cursor.pos);
  emit mouseMoved(m_cursor.pos.toPoint(), distanceToCenter_mm);

  if (m_cursor.leftButtonDown) {
    switch (m_cursorMode) {
    case CursorMode::Default:
      QGraphicsView::mouseMoveEvent(event);
      break;

    case CursorMode::Pan: {
      panMoveEvent(event);
      break;
    }

    case CursorMode::MeasureLine:
      event->accept();

      if (auto *item = dynamic_cast<LineItem *>(m_currItem); item != nullptr)
          [[likely]] {

        const auto line = m_cursor.line();
        const auto dist = computeDistance_mm(line.p1(), line.p2());
        item->setLine(line);

        if (m_currLabelItem != nullptr) {
          m_currLabelItem->setPos(line.center() + QPointF{10, 10});
          m_currLabelItem->setText(QString("%1 mm").arg(dist, 5, 'f', 2));
        }
      }

      break;

    case CursorMode::LabelRect:
      event->accept();
      if (auto *item = dynamic_cast<RectItem *>(m_currItem); item != nullptr)
          [[likely]] {
        item->setRect(m_cursor.rect());
      }
      break;
    case CursorMode::LabelFan:
      event->accept();
      if (auto *item = dynamic_cast<FanItem *>(m_currItem); item != nullptr)
          [[likely]] {

        const auto angle = m_cursor.angle(m_Pixmap.rect());
        // emit error(QString("Fan angle: %1").arg(angle));
        item->setSpanAngle(angle - item->startAngle());
      }
      break;
    }

  } else if (m_cursor.middleButtonDown) {
    // Panning
    panMoveEvent(event);

  } else {
    QGraphicsView::mouseMoveEvent(event);
  }
}

void Canvas::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = false;

    switch (m_cursorMode) {

    case CursorMode::Default:
      QGraphicsView::mouseReleaseEvent(event);
      break;

    case CursorMode::Pan:
      panEndEvent(event);
      break;

    case CursorMode::MeasureLine: {
      // Save
      Annotation anno(m_cursor.line(), m_currItem->color());
      m_annotations->addAnnotation(anno);

    } break;
    case CursorMode::LabelRect: {
      // Save
      Annotation anno(m_cursor.rect(), m_currItem->color());
      m_annotations->addAnnotation(anno);

    } break;
    case CursorMode::LabelFan:

      if (auto *item = dynamic_cast<FanItem *>(m_currItem); item != nullptr)
          [[likely]] {
        Annotation anno(item->arc(), item->color());
        m_annotations->addAnnotation(anno);
      }
      break;
    }

  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = false;
    panEndEvent(event);

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = false;
    QGraphicsView::mouseReleaseEvent(event);
  }
}

void Canvas::wheelEvent(QWheelEvent *event) {
  const auto WHEEL_ZOOM_MODIFIER = Qt::ControlModifier;

  if (event->modifiers().testFlag(WHEEL_ZOOM_MODIFIER)) {
    // modifier + scroll -> Zoom
    event->accept();

    // Calculate the scale factor adjustment
    const double numDegrees = event->angleDelta().y() / 8.0;
    const double numSteps = numDegrees / 15.0;
    const double sensitivity = 0.1;
    const double scaleFactor = 1.0 - numSteps * sensitivity;
    m_scaleFactor = std::max(m_scaleFactor * scaleFactor, m_scaleFactorMin);

    updateTransform();
  } else {
    QGraphicsView::wheelEvent(event);
  }
}

bool Canvas::gestureEvent(QGestureEvent *event) {
  if (QGesture *pinch = event->gesture(Qt::PinchGesture)) {
    pinchTriggered(
        static_cast<QPinchGesture *>(pinch)); // NOLINT(*-static-cast-downcast)
  }
  return true;
}

void Canvas::pinchTriggered(QPinchGesture *gesture) {
  if (gesture->state() == Qt::GestureState::GestureUpdated) {
    const qreal scaleFactor = gesture->scaleFactor();
    m_scaleFactor = std::max(m_scaleFactor * scaleFactor, m_scaleFactorMin);
    updateTransform();
  }
}

void Canvas::panStartEvent(QMouseEvent *event) {
  event->accept();
  m_lastPanPoint = event->pos();

  setCursor(Qt::ClosedHandCursor);
}

void Canvas::panMoveEvent(QMouseEvent *event) {
  event->accept();
  QPointF delta = event->pos() - m_lastPanPoint;
  horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
  verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
  m_lastPanPoint = event->pos();
}

void Canvas::panEndEvent(QMouseEvent *event) { setCursor(Qt::OpenHandCursor); }

void Canvas::updateTransform() {
  // Update transform for the QGraphicsView
  QTransform transform;
  transform.scale(m_scaleFactor, m_scaleFactor);
  setTransform(transform);

  // Update font factor for the annotation text labels
  if (m_currLabelItem != nullptr) {
    auto font = m_currLabelItem->font();
    font.setPointSizeF(16.0 / m_scaleFactor);
    m_currLabelItem->setFont(font);
  }

  // Update zoom factor displayed in the overlay
  m_overlay->setZoom(m_scaleFactor);
}

void Canvas::resizeEvent(QResizeEvent *event) {
  updateMinScaleFactor();

  QGraphicsView::resizeEvent(event);

  m_overlay->resize(viewport()->size());
}

void Canvas::setCursorMode(CursorMode mode) {
  m_cursorMode = mode;

  // Set cursor
  switch (mode) {
  case CursorMode::Pan:
    setCursor(Qt::OpenHandCursor);
    break;
  default:
    setCursor(Qt::CrossCursor);
  }
}

void Canvas::addAnnotationItem(int row) {
  const Annotation &annotation = m_annotations->getAnnotation(row);

  QGraphicsItem *item = [&]() -> QGraphicsItem * {
    switch (annotation.type()) {

    case Annotation::Line: {
      auto *item = new QGraphicsLineItem(
          QLineF(annotation.rect().topLeft(), annotation.rect().bottomRight()));
      item->setPen(QPen(annotation.color()));
      return item;
    }

    case Annotation::Rect: {
      auto *item = new QGraphicsRectItem(annotation.rect());
      item->setPen(QPen(annotation.color()));
      return item;
    }

    case Annotation::Fan: {
      // TODO
      break;
    }
    case Annotation::Polygon: {
      auto *item = new QGraphicsPolygonItem(annotation.polygon());
      item->setPen(QPen(annotation.color()));
      return item;
    }
    }
  }();

  scene()->addItem(item);
  m_annotationItems.append(item);
}

void Canvas::updateAnnotationItem(int row) {
  QGraphicsItem *item = m_annotationItems[row];
  const Annotation &annotation = m_annotations->getAnnotation(row);

  switch (annotation.type()) {
  case Annotation::Line:
    if (auto *lineItem = dynamic_cast<QGraphicsLineItem *>(item);
        lineItem != nullptr) {
      lineItem->setLine(
          QLineF(annotation.rect().topLeft(), annotation.rect().bottomRight()));
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

  case Annotation::Fan:
    // TODO
    // if (auto *fanItem = dynamic_cast<)

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

void Canvas::removeAnnotationItem(int row) {
  QGraphicsItem *item = m_annotationItems.takeAt(row);
  scene()->removeItem(item);
  delete item;
}

// NOLINTEND(*-casting, *-narrowing-conversions)
void Canvas::removeCurrItem() {
  if (m_currItem != nullptr) {
    m_scene->removeItem(m_currItem);
    delete m_currItem;
    m_currItem = nullptr;
  }

  if (m_currLabelItem != nullptr) {
    m_scene->removeItem(m_currLabelItem);
    delete m_currLabelItem;
    m_currLabelItem = nullptr;
  }
}
