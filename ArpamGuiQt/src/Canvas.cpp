#include "Canvas.hpp"
#include "geometryUtils.hpp"
#include <QHBoxLayout>
#include <QImage>
#include <QPainter>
#include <QVBoxLayout>
#include <QtCore>
#include <QtDebug>
#include <QtLogging>
#include <array>
#include <cmath>
#include <tuple>
#include <uspam/timeit.hpp>

Canvas::Canvas(QWidget *parent) : QLabel(parent) {
  setBackgroundRole(QPalette::Base);
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setAlignment(Qt::AlignCenter);
  setMouseTracking(true);

  setFocusPolicy(Qt::StrongFocus);
}

void Canvas::imshow(const cv::Mat &cv_img, double pix2m) {
  QImage qi(cv_img.data, cv_img.cols, cv_img.rows,
            static_cast<int>(cv_img.step), QImage::Format_BGR888);
  this->imshow(QPixmap::fromImage(qi), pix2m);
}

void Canvas::imshow(const QImage &img, double pix2m) {
  this->imshow(QPixmap::fromImage(img), pix2m);
}

void Canvas::imshow(const QPixmap &pixmap, double pix2m) {
  m_pixmap = pixmap;
  m_pix2m = pix2m;

  // clear cached scaled pixmap
  m_pixmapScaled = QPixmap();

  update();
}

// NOLINTBEGIN(*-casting, *-narrowing-conversions)

void Canvas::drawTicks(QPainter *painter) {
  // Update scale bar
  constexpr int m2mm = 1000;
  m_ticks.update(m_pixmapScaled.size(), m2mm * m_pix2m / m_scale);

  // Painting
  m_ticks.draw(painter);
}

// Compute the distance between two points in the scaled pixmap domain
double Canvas::computeDistance_mm(QPointF pt1, QPointF pt2) const {
  const auto distance = geometry::calcMagnitude(pt1 - pt2);
  const auto distance_mm = distance * m_pix2m * 1000;
  return distance_mm;
}

double Canvas::computeDistanceScaled_mm(QPointF pt1, QPointF pt2) const {
  return computeDistance_mm(pt1, pt2) / m_scale;
}

void Canvas::paintEvent(QPaintEvent *event) {
  if (m_pixmap.isNull()) {
    return;
  }

  uspam::TimeIt timeit;
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);

  // Draw scaled pixmap
  {
    // Canvas size
    const auto w = width();
    const auto h = height();

    const auto [pw, ph, scale] = [&] {
      // Pixmap size
      int pw, ph;
      if (m_zoomed) {
        pw = m_zoomRect.width();
        ph = m_zoomRect.height();
      } else {
        pw = m_pixmap.width();
        ph = m_pixmap.height();
      }

      // Calculate scale factor to maintain aspect ratio
      qreal scale = qMin(w / (qreal)pw, h / (qreal)ph);
      return std::tuple{pw, ph, scale};
    }();

    // Calculate the position to center pixmap
    m_offset = QPoint((w - pw * scale) / 2, (h - ph * scale) / 2);

    // Set transformation
    painter.translate(m_offset);

    // if m_pixmapScaled is null OR scale changed, recompute scaled pixmap
    if (m_pixmapScaled.isNull() || scale != m_scale ||
        (m_zoomed && m_zoomTranslated)) {
      m_scale = scale;
      // Rescale everything here!

      transformForward = QTransform()
                             .scale(scale, scale)
                             .translate(-m_zoomRect.left(), -m_zoomRect.top());

      transformBackward = QTransform()
                              .translate(m_zoomRect.left(), m_zoomRect.top())
                              .scale(1 / scale, 1 / scale);

      if (m_zoomed) {
        auto tmp = m_pixmap.copy(m_zoomRect.toRect());
        m_pixmapScaled = tmp.scaled(m_scale * tmp.size(), Qt::KeepAspectRatio);

        // m_pixmapScaled = m_pixmap.transformed(transformForward);
      } else {
        m_pixmapScaled =
            m_pixmap.scaled(m_scale * m_pixmap.size(), Qt::KeepAspectRatio);
      }

      // Update scalebar
      {
        constexpr int m2mm = 1000;
        m_ticks.update(m_pixmapScaled.size(), m2mm * m_pix2m / m_scale);
      }

      // Update annotations
      // m_anno.rescale(m_scale, m_zoomRect.topLeft());
      m_anno.rescale(transformForward);
    }

    // Draw scaled pixmap
    painter.drawPixmap(QPoint{}, m_pixmapScaled);
  }

  // Draw ticks
  m_ticks.draw(&painter);

  // Draw canvas name
  if (!m_name.isNull()) {
    const int margin = 10;

    QRect boundingRect = QRect(QPoint{}, m_pixmapScaled.size());
    boundingRect.adjust(0, 0, -margin, -margin);

    painter.setPen(Qt::white);
    painter.drawText(boundingRect, Qt::AlignRight | Qt::AlignBottom, m_name);
  }

  // Draw existing annotations
  {
    painter.setPen(Qt::white);

    // Draw lines
    {
      painter.drawLines(m_anno.lines.scaled.data(), m_anno.lines.scaled.size());

      for (const auto &line : m_anno.lines.scaled) {
        const auto distance = computeDistanceScaled_mm(line.p1(), line.p2());
        const auto msg = QString("%1 mm").arg(distance);
        const auto textPos = line.p2() + QPointF(5, 5);
        painter.drawText(textPos, msg);
      }

      painter.drawLines(m_anno.lines.whiskers.data(),
                        m_anno.lines.whiskers.size());
    }

    // Draw rects
    {
      painter.drawRects(m_anno.rects.scaled.data(), m_anno.rects.scaled.size());
    }
  }

  // Draw curr annotation
  switch (m_cursorMode) {
  case CursorMode::LineMeasure: {

    if (m_cursor.leftButtonDown) {
      painter.setPen(Qt::white);
      const auto line = m_cursor.getLine();
      painter.drawLine(line);

      const auto distance = computeDistanceScaled_mm(line.p1(), line.p2());
      const auto msg = QString("%1 mm").arg(distance);
      const auto textPos = line.p2() + QPointF(5, 5);
      painter.drawText(textPos, msg);

      const auto whiskers = m_anno.lines.computeLineWhisker(line);
      painter.drawLines(whiskers.data(), whiskers.size());
    }
    break;
  }

  case CursorMode::BoxZoom: {
    if (m_cursor.leftButtonDown) {
      painter.setPen(Qt::white);

      const auto rect = m_cursor.getRect();
      painter.drawRect(rect);
    }

    break;
  }
  }

  // Measure rendering time
  // {
  //   const auto renderTime_ms = timeit.get_ms();
  //   auto msg = QString("Rendering time %1 ms").arg(renderTime_ms);
  //   emit error(msg);
  // }
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

    // Go back in zoom history
    if (m_zoomed) {
      if (m_zoomRectHistory.size() > 1) {
        m_zoomRectHistory.pop_back();
        m_zoomRect = m_zoomRectHistory.back();
        m_zoomed = true;
        m_zoomTranslated = true;
      } else {
        m_zoomRectHistory.clear();
        m_zoomed = false;
        m_zoomTranslated = false;
        m_zoomRect.setTopLeft({0.0, 0.0});
      }
      update();
    }
    break;
  }
  }
}

void Canvas::mousePressEvent(QMouseEvent *event) {
  m_cursor.startPos = event->position() - m_offset;

  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = true;

    // // Only show one annotation on screen for now
    // m_anno.clear();
    // update();

  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = true;

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = true;

    undo();
  }
}

void Canvas::mouseMoveEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  m_cursor.currPos = event->position() - m_offset;
  QPointF pos = m_cursor.currPos / m_scale;
  if (m_zoomed) {
    pos += m_zoomRect.topLeft();
  }
  m_cursor.currPosOrigal = pos;

  // [px] Compute distance to center
  const QPointF center(m_pixmap.width() / 2.0, m_pixmap.height() / 2.0);
  auto distanceToCenter_mm = computeDistance_mm(center, pos);
  emit mouseMoved(pos.toPoint(), distanceToCenter_mm);

  if (m_cursor.leftButtonDown) {
    // draw annotation currently drawing
    update();

  } else if (m_cursor.middleButtonDown && m_zoomed) {
    // Move in Zoomed mode
    const auto displacement = (m_cursor.startPos - m_cursor.currPos) / m_scale;
    m_cursor.startPos = m_cursor.currPos;

    // Advanced translate - clip to m_pixmap boundary
    QRectF bound(QPointF{0, 0}, m_pixmap.size());
    const auto zoomRectTranslated =
        geometry::translateBounded(m_zoomRect, displacement, bound);

    if (zoomRectTranslated != m_zoomRect) {
      m_zoomRect = zoomRectTranslated;
      m_zoomTranslated = true;
      update();
    }
  }
}

void Canvas::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = false;

    switch (m_cursorMode) {

    case CursorMode::LineMeasure: {
      // Save line
      const auto lineScaled = m_cursor.getLine();
      // m_anno.lines.addScaled(lineScaled, m_scale, m_zoomRect.topLeft());
      m_anno.lines.addScaled(lineScaled, transformBackward);
      break;
    }

    case CursorMode::BoxZoom: {
      const auto rectScaled = m_cursor.getRect();
      const QRectF rect = transformBackward.mapRect(rectScaled);

      constexpr int RECT_MIN_SIZE = 10;
      if (rect.width() > RECT_MIN_SIZE && rect.height() > RECT_MIN_SIZE) {
        // Save rect to annotations
        // m_anno.rects.push_back(rect);
        // m_anno.rectsScaled.push_back(rectScaled);

        // Set Zoom
        m_zoomRectHistory.push_back(rect);
        m_zoomRect = rect;
        m_zoomed = true;
      }
    }
    }

    update();

  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = false;
    m_zoomRectHistory.push_back(m_zoomRect);

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = false;
  }
}

void Canvas::keyPressEvent(QKeyEvent *event) {}
// NOLINTEND(*-casting, *-narrowing-conversions)
