#include "Annotation/AnnotationView.hpp"
#include "Annotation/Annotation.hpp"
#include <QHBoxLayout>
#include <QMenu>
#include <QModelIndex>
#include <QVBoxLayout>

namespace annotation {

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

void AnnotationView::setModel(AnnotationModel *model) {
  m_model = model;
  m_tableView->setModel(m_model);

  // Select rows
  m_tableView->setSelectionBehavior(QAbstractItemView::SelectRows);

  // Context menu for deletion
  m_tableView->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(m_tableView, &QTableView::customContextMenuRequested,
          [this](const QPoint &pos) {
            QMenu contextMenu;
            QAction deleteAction("Delete Row", &contextMenu);
            contextMenu.addAction(&deleteAction);

            if (const auto index = m_tableView->indexAt(pos); index.isValid()) {
              connect(&deleteAction, &QAction::triggered,
                      [this, index]() { m_model->removeRow(index.row()); });
              contextMenu.exec(m_tableView->viewport()->mapToGlobal(pos));
            }
          });
}

} // namespace annotation