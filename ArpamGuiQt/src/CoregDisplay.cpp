#include "CoregDisplay.hpp"

#include <QHBoxLayout>
#include <QPushButton>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent), canvasLeft(new Canvas(this)),
      canvasRight(new Canvas(this)) {

  auto *layout = new QHBoxLayout(this);
  layout->addWidget(canvasLeft);
  layout->addWidget(canvasRight);
  // layout->addWidget(new QPushButton);
}