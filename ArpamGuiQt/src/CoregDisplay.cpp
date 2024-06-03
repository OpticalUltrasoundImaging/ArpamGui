#include "CoregDisplay.hpp"

#include <QHBoxLayout>
#include <QPushButton>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent), canvas1(new Canvas(this)), canvas2(new Canvas(this)) {

  auto *layout = new QHBoxLayout(this);
  layout->addWidget(canvas1);
  layout->addWidget(canvas2);
  // layout->addWidget(new QPushButton);
}