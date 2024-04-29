#pragma once

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
  void imshow(const cv::Mat &cv_img);
  void imshow(const QImage &img);
  void imshow(const QPixmap &pixmap);
};