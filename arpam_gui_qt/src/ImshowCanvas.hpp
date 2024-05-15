#pragma once

#include <QImage>
#include <QLayout>
#include <QString>
#include <QtWidgets>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

class ImshowCanvas : public QLabel {
  Q_OBJECT
public:
  explicit ImshowCanvas(QWidget *parent = nullptr);

  void setName(QString name) { m_name = name; }

public slots:
  void imshow(const cv::Mat &cv_img, double pix2m);
  void imshow(const QImage &img, double pix2m);
  void imshow(const QPixmap &pixmap, double pix2m);

signals:
  void error(QString err);

  // Signal emitted on mouseMoveEvent. Position is converted to the pixmap
  // domain
  void mouseMoved(QPoint pos, double depth_mm);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;

private:
  // Original pixmap.
  QPixmap m_pixmap;

  // Cache of scaled pixmap
  QPixmap m_pixmapScaled;

  // [m] Factor converting pixel to meters
  double m_pix2m{};

  // Size factor for m_pixmap to maintain aspect ratio
  double m_scale{};

  // Offset of displayed scaled m_pixmap to keep center.
  // m_offset is updated on every paintEvent so it should never be zero
  // when a m_pixmap is present.
  QPoint m_offset{};

  QString m_name;
};