#pragma once

#include "Canvas.hpp"
#include <QLabel>
#include <QWidget>

class CoregDisplay : public QWidget {
  Q_OBJECT
public:
  CoregDisplay(QWidget *parent = nullptr);

  Canvas *canvas1;
  Canvas *canvas2;
};