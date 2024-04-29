#include "ImshowCanvas.hpp"
#include <QHBoxLayout>
#include <QImage>
#include <QtCore>
#include <QtLogging>

ImshowCanvas::ImshowCanvas(QWidget *parent) : QLabel(parent) {
  this->setBackgroundRole(QPalette::Base);
  QSizePolicy sizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  this->setSizePolicy(sizePolicy);
}

void ImshowCanvas::imshow(const cv::Mat &cv_img) {
  QImage qi(cv_img.data, cv_img.cols, cv_img.rows,
            static_cast<int>(cv_img.step), QImage::Format_BGR888);
  this->imshow(QPixmap::fromImage(qi));
  // qDebug("ImshowCanvas::imshow(cv::Mat) called");
}

void ImshowCanvas::imshow(const QImage &img) {
  this->setPixmap(QPixmap::fromImage(img));
  // qDebug("ImshowCanvas::imshow(QPixmap) called");
}

void ImshowCanvas::imshow(const QPixmap &pixmap) {
  // this->setPixmap(pixmap.scaled(this->size(), Qt::KeepAspectRatio,
  //                               Qt::SmoothTransformation));
  this->setPixmap(pixmap);
  // qDebug("ImshowCanvas::imshow(QPixmap) called");
}