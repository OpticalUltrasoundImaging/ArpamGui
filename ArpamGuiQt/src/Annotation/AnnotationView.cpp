#include "Annotation/AnnotationView.hpp"
#include "Annotation/Annotation.hpp"
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QMenu>
#include <QModelIndex>
#include <QPushButton>
#include <QVBoxLayout>
#include <qnamespace.h>
#include <qpushbutton.h>
#include <ranges>

namespace annotation {

namespace details {

AnnotationTableView::AnnotationTableView(QWidget *parent) : QTableView(parent) {
  // Select rows
  setSelectionBehavior(QAbstractItemView::SelectRows);

  // Context menu for deletion
  setContextMenuPolicy(Qt::CustomContextMenu);
  connect(this, &QTableView::customContextMenuRequested,
          [this](const QPoint &pos) {
            QMenu contextMenu;
            QAction deleteAction("Delete Row", &contextMenu);
            contextMenu.addAction(&deleteAction);

            if (const auto index = indexAt(pos); index.isValid()) {
              connect(&deleteAction, &QAction::triggered,
                      [this, index]() { model()->removeRow(index.row()); });
              contextMenu.exec(viewport()->mapToGlobal(pos));
            }
          });
}

void AnnotationTableView::keyPressEvent(QKeyEvent *event) {
  switch (event->key()) {
  case Qt::Key_Delete:
  case Qt::Key_Backspace:
    for (const auto &idx :
         selectionModel()->selectedRows() | std::views::reverse) {
      model()->removeRow(idx.row());
    }
    break;
  default:
    QTableView::keyPressEvent(event);
  }
}

} // namespace details

AnnotationView::AnnotationView(QWidget *parent)
    : QWidget(parent), m_tableView(new details::AnnotationTableView) {

  // UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  {
    auto *hlayout = new QHBoxLayout;
    vlayout->addLayout(hlayout);
  }

  vlayout->addWidget(m_tableView);

  {
    auto *btn = new QPushButton("Save");
    vlayout->addWidget(btn);
    connect(btn, &QPushButton::clicked,
            [this] { m_model->saveToFile("frame.json"); });
  }
}

void AnnotationView::setModel(AnnotationModel *model) {
  m_model = model;
  m_tableView->setModel(m_model);
}

} // namespace annotation