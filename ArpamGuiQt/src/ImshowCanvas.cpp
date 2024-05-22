#include "ImshowCanvas.hpp"
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
#include <uspam/timeit.hpp>

ImshowCanvas::ImshowCanvas(QWidget *parent) : QLabel(parent) {
  setBackgroundRole(QPalette::Base);
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setAlignment(Qt::AlignCenter);
  setMouseTracking(true);
}

void ImshowCanvas::imshow(const cv::Mat &cv_img, double pix2m) {
  QImage qi(cv_img.data, cv_img.cols, cv_img.rows,
            static_cast<int>(cv_img.step), QImage::Format_BGR888);
  this->imshow(QPixmap::fromImage(qi), pix2m);
}

void ImshowCanvas::imshow(const QImage &img, double pix2m) {
  this->imshow(QPixmap::fromImage(img), pix2m);
}

void ImshowCanvas::imshow(const QPixmap &pixmap, double pix2m) {
  m_pixmap = pixmap;
  m_pix2m = pix2m;

  // clear cached scaled pixmap
  m_pixmapScaled = QPixmap();

  this->update();
}

// NOLINTBEGIN(*-casting, *-narrowing-conversions)

void ImshowCanvas::drawTicks(QPainter *painter) {
  // Update scale bar
  constexpr int m2mm = 1000;
  m_ticks.update(m_pixmapScaled.size(), m2mm * m_pix2m / m_scale);

  // Painting
  m_ticks.draw(painter);
}

// Compute the distance between two points in the scaled pixmap domain
double ImshowCanvas::computeDistance_mm(QPointF pt1, QPointF pt2) const {
  const auto distance = geometry::calcMagnitude(pt1 - pt2);
  const auto distance_mm = distance * m_pix2m * 1000;
  return distance_mm;
}

double ImshowCanvas::computeDistanceScaled_mm(QPointF pt1, QPointF pt2) const {
  return computeDistance_mm(pt1, pt2) / m_scale;
}

void ImshowCanvas::paintEvent(QPaintEvent *event) {
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
    // Pixmap size
    const auto pw = m_pixmap.width();
    const auto ph = m_pixmap.height();

    // Calculate scale factor to maintain aspect ratio
    qreal scale = qMin(w / (qreal)pw, h / (qreal)ph);

    // Calculate the position to center pixmap
    m_offset = QPoint((w - pw * scale) / 2, (h - ph * scale) / 2);

    // Set transformation
    painter.translate(m_offset);

    // if m_pixmapScaled is null OR scale changed, recompute scaled pixmap
    if (m_pixmapScaled.isNull() || scale != m_scale) {
      m_scale = scale;
      // Rescale everything here!

      m_pixmapScaled =
          m_pixmap.scaled(m_scale * m_pixmap.size(), Qt::KeepAspectRatio);

      // Update scalebar
      {
        constexpr int m2mm = 1000;
        m_ticks.update(m_pixmapScaled.size(), m2mm * m_pix2m / m_scale);
      }

      // Update annotations
      m_anno.rescale(scale);
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
      painter.drawLines(m_anno.linesScaled.data(), m_anno.linesScaled.size());

      for (const auto &line : m_anno.linesScaled) {
        const auto distance = computeDistanceScaled_mm(line.p1(), line.p2());
        const auto msg = QString("%1 mm").arg(distance);
        const auto textPos = line.p2() + QPointF(5, 5);
        painter.drawText(textPos, msg);
      }

      painter.drawLines(m_anno.lineWhiskers.data(), m_anno.lineWhiskers.size());
    }

    // Draw rects
    { painter.drawRects(m_anno.rectsScaled.data(), m_anno.rectsScaled.size()); }
  }

  // Draw curr annotation
  switch (m_cursorType) {
  case CursorType::LineMeasure: {

    if (m_cursor.leftButtonDown) {
      painter.setPen(Qt::white);
      const auto line = m_cursor.getLine();
      painter.drawLine(line);

      const auto distance = computeDistanceScaled_mm(line.p1(), line.p2());
      const auto msg = QString("%1 mm").arg(distance);
      const auto textPos = line.p2() + QPointF(5, 5);
      painter.drawText(textPos, msg);

      const auto whiskers = m_anno.computeLineWhisker(line);
      painter.drawLines(whiskers.data(), whiskers.size());
    }
    break;
  }

  case CursorType::BoxZoom: {
    if (m_cursor.leftButtonDown) {
      painter.setPen(Qt::white);

      const auto rect = m_cursor.getRect();
      painter.drawRect(rect);
    }

    break;
  }
  }

  // Measure rendering time
  {
    const auto renderTime_ms = timeit.get_ms();
    auto msg = QString("Rendering time %1 ms").arg(renderTime_ms);
    emit error(msg);
  }
}

void ImshowCanvas::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = true;
    m_cursor.startPos = event->position() - m_offset;

    // Only show one annotation on screen for now
    m_anno.clear();

  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = true;

  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = true;

    if (!m_anno.empty()) {
      m_anno.clear();

      repaint();
    }
  }
}

void ImshowCanvas::mouseMoveEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  if (m_pixmap.isNull()) {
    return;
  }

  m_cursor.currPos = event->position() - m_offset;
  {
    const auto pos = (m_cursor.currPos / m_scale);

    // [px] Compute distance to center
    const QPointF center(m_pixmap.width() / 2.0, m_pixmap.height() / 2.0);
    auto distance_mm = computeDistance_mm(center, pos);

    emit mouseMoved(pos.toPoint(), distance_mm);
  }

  // draw annotation currently drawing
  if (m_cursor.leftButtonDown) {
    update();
  }
}

void ImshowCanvas::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursor.leftButtonDown = false;

    switch (m_cursorType) {
    case CursorType::LineMeasure: {
      // Save line
      const auto lineScaled = m_cursor.getLine();
      const QLineF line(lineScaled.p1() / m_scale, lineScaled.p2() / m_scale);
      m_anno.linesScaled.push_back(lineScaled);
      m_anno.lines.push_back(line);

      const auto whiskers = m_anno.computeLineWhisker(lineScaled);
      m_anno.lineWhiskers.push_back(whiskers[0]);
      m_anno.lineWhiskers.push_back(whiskers[1]);
      break;
    }
    case CursorType::BoxZoom: {
      const auto rectScaled = m_cursor.getRect();
      const QRectF rect(rectScaled.x() / m_scale, rectScaled.y() / m_scale,
                        rectScaled.width() / m_scale,
                        rectScaled.height() / m_scale);

      m_anno.rects.push_back(rect);
      m_anno.rectsScaled.push_back(rectScaled);
    }
    }

  } else if (event->button() == Qt::MiddleButton) {
    m_cursor.middleButtonDown = false;
  } else if (event->button() == Qt::RightButton) {
    m_cursor.rightButtonDown = false;
  }
}

// NOLINTEND(*-casting, *-narrowing-conversions)
