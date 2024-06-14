#include "CanvasAnnotationView.hpp"
#include <QHBoxLayout>
#include <QVBoxLayout>

AnnotationView::AnnotationView(QWidget *parent)
    : QWidget(parent), m_tableView(new QTableView) {

  // UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  {
    auto *hlayout = new QHBoxLayout;
    vlayout->addLayout(hlayout);
  }

  { vlayout->addWidget(m_tableView); }
}
