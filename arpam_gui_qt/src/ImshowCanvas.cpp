#include "ImshowCanvas.hpp"
#include <QImage>
#include <QPainter>
#include <QtCore>
#include <QtDebug>
#include <QtLogging>
#include <uspam/timeit.hpp>

namespace {

auto clipPoint(const QPointF &point, const QPointF &minPoint,
               const QPointF &maxPoint) {
  auto clippedX = std::max(minPoint.x(), std::min(maxPoint.x(), point.x()));
  auto clippedY = std::max(minPoint.y(), std::min(maxPoint.y(), point.y()));
  return QPointF(clippedX, clippedY);
}

auto clipPoint(const QPoint &point, const QPoint &minPoint,
               const QPoint &maxPoint) {
  auto clippedX = std::max(minPoint.x(), std::min(maxPoint.x(), point.x()));
  auto clippedY = std::max(minPoint.y(), std::min(maxPoint.y(), point.y()));
  return QPoint(clippedX, clippedY);
}

} // namespace

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

void ImshowCanvas::drawScaleBar(QPainter *painter) {
  // Update scale bar
  constexpr int m2mm = 1000;
  m_scalebar.update(m_pixmapScaled.size(), m2mm * m_pix2m / m_scale);

  // Painting
  m_scalebar.draw(painter);
}

double ImshowCanvas::computeDistance_mm(QPointF pt1, QPointF pt2) {
  const auto d = pt1 - pt2;
  const auto distance = std::sqrt(d.x() * d.x() + d.y() * d.y());
  const auto distance_mm = distance * m_pix2m * 1000;
  return distance_mm;
}

void ImshowCanvas::paintEvent(QPaintEvent *event) {
  // QLabel::paintEvent(event);

  if (!m_pixmap.isNull()) {
    uspam::TimeIt timeit;
    QPainter painter(this);
    // painter.setRenderHint(QPainter::Antialiasing);

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

        m_pixmapScaled =
            m_pixmap.scaled(m_scale * m_pixmap.size(), Qt::KeepAspectRatio);

        // update scalebar
        {
          constexpr int m2mm = 1000;
          m_scalebar.update(m_pixmapScaled.size(), m2mm * m_pix2m / m_scale);
        }
      }

      // Draw scaled pixmap
      painter.drawPixmap(QPoint{}, m_pixmapScaled);
    }

    // Draw scalebar
    {
      uspam::TimeIt timeit;
      m_scalebar.draw(&painter);
      const auto elapsed = timeit.get_ms();
      // emit error(QString("drawScaleBar took %1").arg(elapsed));
    }

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
      painter.drawLines(m_anno.lines.data(), m_anno.lines.size());

      for (const auto &line : m_anno.lines) {
        const auto distance = computeDistance_mm(line.p1(), line.p2());
        const auto msg = QString("%1 mm").arg(distance);
        const auto textPos = line.p2() + QPointF(5, 5);
        painter.drawText(textPos, msg);
      }
    }

    // Draw curr annotation
    {
      if (m_cursorState.leftButtonDown) {
        painter.setPen(Qt::white);
        const auto line = m_cursorState.currLine();
        painter.drawLine(line);

        const auto distance = computeDistance_mm(line.p1(), line.p2());
        const auto msg = QString("%1 mm").arg(distance);
        const auto textPos = line.p2() + QPointF(5, 5);
        painter.drawText(textPos, msg);
      }
    }

    // Measure rendering time
    const auto renderTime_ms = timeit.get_ms();
    // emit error(QString("Rendering time %1 ms").arg(renderTime_ms));
  }
}

void ImshowCanvas::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursorState.leftButtonDown = true;

    // Only show one line on screen for now
    m_anno.lines.clear();

    const auto pos = event->position() - m_offset;
    m_cursorState.startPos = pos;
  } else if (event->button() == Qt::MiddleButton) {
    m_cursorState.middleButtonDown = true;
  } else if (event->button() == Qt::RightButton) {
    m_cursorState.rightButtonDown = true;

    if (!m_anno.lines.empty()) {
      m_anno.lines.clear();
      repaint();
    }
  }
}

void ImshowCanvas::mouseMoveEvent(QMouseEvent *event) {
  // Compute position in the pixmap domain
  if (m_pixmap.isNull()) {
    return;
  }

  m_cursorState.currPos = event->position() - m_offset;
  {
    const QPoint pos = (m_cursorState.currPos / m_scale).toPoint();

    // Compute position offset from center of the pixmap
    const auto dx = m_pixmap.width() / 2 - pos.x();
    const auto dy = m_pixmap.height() / 2 - pos.y();

    // [px] Compute distance
    const qreal distance = std::sqrt(dx * dx + dy * dy);
    const qreal distance_mm = distance * m_pix2m * 1000;

    emit mouseMoved(pos, distance_mm);
  }

  // draw annotation currently drawing
  if (m_cursorState.leftButtonDown) {
    update();
  }
}

void ImshowCanvas::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    m_cursorState.leftButtonDown = false;

    m_anno.lines.push_back(m_cursorState.currLine());

  } else if (event->button() == Qt::MiddleButton) {
    m_cursorState.middleButtonDown = false;
  } else if (event->button() == Qt::RightButton) {
    m_cursorState.rightButtonDown = false;
  }
}

// NOLINTEND(*-casting, *-narrowing-conversions)