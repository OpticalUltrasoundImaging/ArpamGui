#pragma once

#include "ImshowCanvas.hpp"
#include <QLabel>
#include <QWidget>

class CoregDisplay : public QWidget {
  Q_OBJECT
public:
  CoregDisplay(QWidget *parent = nullptr);

  ImshowCanvas *canvas1;
  ImshowCanvas *canvas2;
};