#include "Canvas.hpp"
#include "CanvasAnnotations.hpp"
#include "geometryUtils.hpp"
#include <QGestureEvent>
#include <QHBoxLayout>
#include <QImage>
#include <QPainter>
#include <QPinchGesture>
#include <QVBoxLayout>
#include <QtCore>
#include <QtDebug>
#include <QtLogging>
#include <array>
#include <cmath>
#include <qgraphicsview.h>
#include <tuple>
#include <uspam/timeit.hpp>

Canvas::Canvas(QWidget *parent)
    : QGraphicsView(parent), m_scene(new QGraphicsScene) {
  setBackgroundRole(QPalette::Base);
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setAlignment(Qt::AlignCenter);

  // Enable mouse tracking
  setMouseTracking(true);
  setFocusPolicy(Qt::StrongFocus);

  // Enable pinch gesture handling
  // setDragMode(DragMode::ScrollHandDrag);
  viewport()->setAttribute(Qt::WA_AcceptTouchEvents);
  grabGesture(Qt::PinchGesture);

  // Graphics rendering parameters
  setRenderHint(QPainter::Antialiasing);
  setRenderHint(QPainter::SmoothPixmapTransform);
  setScene(m_scene);
}

void Canvas::imshow(const cv::Mat &cv_img, double pix2m) {
  QImage qi(cv_img.data, cv_img.cols, cv_img.rows,
            static_cast<int>(cv_img.step), QImage::Format_BGR888);
  this->imshow(QPixmap::fromImage(qi), pix2m);
}

void Canvas::scaleToSize() {
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
  m_scaleFactor = scale;
  m_scaleFactorMin = scale;

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
  if (m_Pixmap.isNull()) {
    return;
  }

  uspam::TimeIt timeit;

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
    const auto renderTime_ms = timeit.get_ms();
    auto msg = QString("Rendering time %1 ms").arg(renderTime_ms);
    emit error(msg);
  }
}

void Canvas::undo() {
  switch (m_cursorMode) {

  case CursorMode::LineMeasure: {
    if (!m_anno.lines.empty()) {
      m_anno.lines.pop();
      update();
    }
    break;
  }

  case CursorMode::BoxZoom: {
  }

  default:
    break;
  }
}

void Canvas::mousePressEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  m_cursor.startPos = m_PixmapItem->mapFromScene(mapToScene(event->pos()));

  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = true;

    switch (m_cursorMode) {
    case (CursorMode::LineMeasure): {
      const auto line = m_cursor.getLine();

      delete m_currLineItem;
      m_currLineItem = m_scene->addLine(line, QPen(Qt::white));
    }
    case CursorMode::BoxZoom: {
      break;
    }
    }

  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = true;

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = true;

    undo();
  }
}

void Canvas::mouseMoveEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  m_cursor.pos = m_PixmapItem->mapFromScene(mapToScene(event->pos()));

  // [px] Compute distance to center
  const QPointF center(m_Pixmap.width() / 2.0, m_Pixmap.height() / 2.0);
  auto distanceToCenter_mm = computeDistance_mm(center, m_cursor.pos);
  emit mouseMoved(m_cursor.pos.toPoint(), distanceToCenter_mm);

  if (m_cursor.leftButtonDown) {
    switch (m_cursorMode) {
    case (CursorMode::LineMeasure): {
      if (m_currLineItem != nullptr) {
        m_currLineItem->setLine(m_cursor.getLine());
      }
      break;
    }
    case CursorMode::BoxZoom: {
      break;
    }
    }
  }
}

void Canvas::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = false;

    switch (m_cursorMode) {

    case CursorMode::LineMeasure: {
      // Save line
      const auto line = m_cursor.getLine();
      m_anno.lines.add(line);
      break;
    }
    case CursorMode::BoxZoom: {
      break;
    }
    }

  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = false;

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = false;
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

void Canvas::updateTransform() {
  QTransform transform;
  transform.scale(m_scaleFactor, m_scaleFactor);
  setTransform(transform);
}
// NOLINTEND(*-casting, *-narrowing-conversions)
