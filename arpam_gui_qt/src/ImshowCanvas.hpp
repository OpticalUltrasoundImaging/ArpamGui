#pragma once

#include <QImage>
#include <QLayout>
#include <QtWidgets>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

class ImshowCanvas : public QLabel {
  Q_OBJECT
public:
  explicit ImshowCanvas(QWidget *parent = nullptr);

public slots:
  void imshow(const cv::Mat &cv_img, double pix2m);
  void imshow(const QImage &img, double pix2m);
  void imshow(const QPixmap &pixmap, double pix2m);

signals:
  void error(QString err);

  // Signal emitted on mouseMoveEvent. Position is converted to the pixmap
  // domain
  void mouseMoved(QPoint pos);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;

private:
  QPixmap m_pixmap;
  double m_pix2m{};
  double m_scale{};  // Size factor for m_pixmap to maintain aspect ratio
  QPoint m_offset{}; // Offset of displayed scaled m_pixmap to keep center
};