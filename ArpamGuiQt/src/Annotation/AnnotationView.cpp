#include "Annotation/AnnotationView.hpp"
#include <QAction>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QMenu>
#include <QMessageBox>
#include <QModelIndex>
#include <QPushButton>
#include <QVBoxLayout>
#include <Qt>
#include <ranges>
#include <set>

namespace annotation {

namespace details {

AnnotationTableView::AnnotationTableView(QWidget *parent) : QTableView(parent) {
  // Select rows
  setSelectionBehavior(QAbstractItemView::SelectRows);

  setAcceptDrops(true);
  setDragEnabled(true);
  setDropIndicatorShown(true);
  setDragDropMode(
      QAbstractItemView::DragDrop); // Allows drag and drop within the view

  auto *deleteAction = new QAction("Delete", this);
  deleteAction->setShortcut(QKeySequence::Delete);
  connect(deleteAction, &QAction::triggered, this,
          &AnnotationTableView::deleteSelectedRows);

  auto *copyAction = new QAction("Copy", this);
  copyAction->setShortcut(QKeySequence::Copy);
  connect(copyAction, &QAction::triggered, this,
          &AnnotationTableView::copySelectedItems);
  addAction(copyAction);

  auto *pasteAction = new QAction("Paste", this);
  pasteAction->setShortcut(QKeySequence::Paste);
  connect(pasteAction, &QAction::triggered, this,
          &AnnotationTableView::pasteItems);
  addAction(pasteAction);

  // Context menu for deletion
  setContextMenuPolicy(Qt::CustomContextMenu);
  connect(this, &QTableView::customContextMenuRequested,
          [this, deleteAction, copyAction, pasteAction](const QPoint &pos) {
            if (const auto index = indexAt(pos); index.isValid()) {
              QMenu contextMenu;
              contextMenu.addAction(deleteAction);
              contextMenu.addAction(copyAction);
              contextMenu.addAction(pasteAction);

              contextMenu.exec(viewport()->mapToGlobal(pos));
            }
          });
}

void AnnotationTableView::keyPressEvent(QKeyEvent *event) {
  switch (event->key()) {
  case Qt::Key_Delete:
  case Qt::Key_Backspace:
    deleteSelectedRows();
    break;
  default:
    QTableView::keyPressEvent(event);
  }
}

void AnnotationTableView::deleteSelectedRows() {
  for (const auto &idx :
       selectionModel()->selectedRows() | std::views::reverse) {
    model()->removeRows(idx.row(), 1);
  }
};

void AnnotationTableView::copySelectedItems() const {
  QItemSelectionModel *selectionModel = this->selectionModel();
  QModelIndexList indexes = selectionModel->selectedIndexes();

  if (indexes.isEmpty()) {
    return;
  }

  // Use the model's mimeData() to package the selected indexes
  if (QMimeData *mimeData = model()->mimeData(indexes); mimeData) {
    QClipboard *clipboard = QGuiApplication::clipboard();
    clipboard->setMimeData(mimeData);
  }
}

void AnnotationTableView::pasteItems() {
  QClipboard *clipboard = QGuiApplication::clipboard();
  const QMimeData *mimeData = clipboard->mimeData();

  // JSON impl
  model()->dropMimeData(mimeData, Qt::DropAction::MoveAction,
                        model()->rowCount(), {}, {});
}
} // namespace details

AnnotationView::AnnotationView(QWidget *parent)
    : QWidget(parent), m_tableView(new details::AnnotationTableView) {

  setAcceptDrops(true);

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
    connect(btn, &QPushButton::clicked, [this] {
      QMessageBox::information(
          this, "Message",
          "Annotations are automatically saved when switching between frames.");
    });
  }
}

void AnnotationView::setModel(AnnotationModel *model) {
  m_model = model;
  m_tableView->setModel(m_model);
}

} // namespace annotation
