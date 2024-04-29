#include "CoregDisplay.hpp"

#include <QHBoxLayout>
#include <QPushButton>

CoregDisplay::CoregDisplay(QWidget *parent)
    : QWidget(parent), canvas1(new ImshowCanvas(this)),
      canvas2(new ImshowCanvas(this)) {

  auto *layout = new QHBoxLayout(this);

  // layout->addWidget(canvas1, 0, Qt::AlignCenter);
  // layout->addWidget(canvas2, 0, Qt::AlignCenter);
  layout->addWidget(canvas1);
  layout->addWidget(canvas2);

  // layout->addWidget(new QPushButton);
}