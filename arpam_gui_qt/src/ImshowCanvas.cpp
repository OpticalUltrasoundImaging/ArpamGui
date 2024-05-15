#include "ImshowCanvas.hpp"
#include <QHBoxLayout>
#include <QImage>
#include <QPainter>
#include <QtCore>
#include <QtDebug>
#include <QtLogging>
#include <uspam/timeit.hpp>

ImshowCanvas::ImshowCanvas(QWidget *parent) : QLabel(parent) {
  this->setBackgroundRole(QPalette::Base);
  this->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  this->setAlignment(Qt::AlignCenter);
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

  this->update();
}

void drawScaleBar(QPainter *painter, int x, int y, int pw, int ph, double pix2m,
                  double displayScale) {
  // Bar parameters
  const int barLength = 20;
  const int barLengthBig = 30;

  // Draw divisions in increments of 1 mm
  const double pix2mm = pix2m * 1000; // [mm]
  // const int numDivisions = 10;
  // const int divisionHeight = barHeight / numDivisions;
  const double divisionSize = 1 / pix2mm;

  painter->save();
  painter->setBrush(Qt::black);

  // Draw vertical scale bar
  {
    const int barWidth = barLength;
    const int barWidthBig = barLengthBig;
    const int barHeight = static_cast<int>(ph * displayScale);

    // Draw scale bar background
    painter->setPen(Qt::black);
    painter->drawRect(0, 0, barWidth, barHeight);

    const int numDivisions = static_cast<int>(ph / (2 * divisionSize));

    // Draw divisions (vertical scale)
    for (int i = 0; i < numDivisions; ++i) {
      // y Position of the current bar
      const auto bar1y =
          static_cast<int>(((double)ph / 2 + i * divisionSize) * displayScale);
      const auto bar2y =
          static_cast<int>(((double)ph / 2 - i * divisionSize) * displayScale);

      // painter->drawLine(0, i * divisionHeight, barWidth, i * divisionHeight);
      if (i % 5 != 0) {
        painter->setPen(Qt::gray);

        painter->drawLine(0, bar1y, barWidth, bar1y);
        painter->drawLine(0, bar2y, barWidth, bar2y);
      } else { // Bigger bars for increments of 5mm
        painter->setPen(Qt::white);

        painter->drawLine(0, bar1y, barWidthBig, bar1y);
        painter->drawLine(0, bar2y, barWidthBig, bar2y);

        painter->drawText(barWidthBig + 5, bar1y + 5, QString::number(i));
        painter->drawText(barWidthBig + 5, bar2y + 5, QString::number(i));
      }
    }
  }

  // Draw horizontal scale bar
  {
    const int barWidth = static_cast<int>(pw * displayScale);
    const int barHeight = barLength;
    const int barHeightBig = barLengthBig;

    // Draw scale bar background
    painter->setPen(Qt::black);
    painter->drawRect(0, 0, barWidth, barHeight);

    const int numDivisions = static_cast<int>(pw / (2 * divisionSize));

    // Draw divisions (vertical scale)
    for (int i = 0; i < numDivisions; ++i) {
      // y Position of the current bar
      const auto bar1x =
          static_cast<int>(((double)pw / 2 + i * divisionSize) * displayScale);
      const auto bar2x =
          static_cast<int>(((double)pw / 2 - i * divisionSize) * displayScale);

      // painter->drawLine(0, i * divisionHeight, barWidth, i * divisionHeight);
      if (i % 5 != 0) {
        painter->setPen(Qt::gray);

        painter->drawLine(bar1x, 0, bar1x, barHeight);
        painter->drawLine(bar2x, 0, bar2x, barHeight);
      } else { // Bigger bars for increments of 5mm
        painter->setPen(Qt::white);

        painter->drawLine(bar1x, 0, bar1x, barHeightBig);
        painter->drawLine(bar2x, 0, bar2x, barHeightBig);

        painter->drawText(bar1x - 5, barHeightBig + 15, QString::number(i));
        painter->drawText(bar2x - 5, barHeightBig + 15, QString::number(i));
      }
    }
  }
  painter->restore();
}

void ImshowCanvas::paintEvent(QPaintEvent *event) {
  // QLabel::paintEvent(event);

  if (!m_pixmap.isNull()) {
    uspam::TimeIt timeit;

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Canvas size
    const auto w = width();
    const auto h = height();
    // Pixmap size
    const auto pw = m_pixmap.width();
    const auto ph = m_pixmap.height();

    // Calculate scale factor to maintain aspect ratio
    qreal scale = qMin(w / (qreal)pw, h / (qreal)ph);

    // Calculate the position to center pixmap
    const int xOffset = (w - pw * scale) / 2;
    const int yOffset = (h - ph * scale) / 2;

    // Set transformation
    painter.translate(xOffset, yOffset);

    painter.save();
    painter.scale(scale, scale);
    // Draw the pixmap centered
    painter.drawPixmap(0, 0, m_pixmap);
    painter.restore();

    // Draw scale bars
    drawScaleBar(&painter, 0, 0, pw, ph, m_pix2m, scale);

    // drawScaleBar(&painter, xOffset, yOffset, scaledPixmap.width(),
    //              scaledPixmap.height(),
    //              m_pix2m * scaledPixmap.width() / m_pixmap.width());

    emit error(QString("Rendering time %1 ms").arg(timeit.get_ms()));
  }
}
