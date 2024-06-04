#pragma once

#include "Canvas.hpp"
#include <QLabel>
#include <QWidget>

class CoregDisplay : public QWidget {
  Q_OBJECT
public:
  CoregDisplay(QWidget *parent = nullptr);

signals:

public slots:
  void setCursorMode(Canvas::CursorMode mode) {
    canvasLeft->setCursorMode(mode);
    canvasRight->setCursorMode(mode);
  }

private:
  Canvas *canvasLeft;
  Canvas *canvasRight;
};