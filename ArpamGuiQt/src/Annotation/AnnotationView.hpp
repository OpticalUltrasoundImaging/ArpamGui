#pragma once

#include "Annotation/AnnotationModel.hpp"
#include <QClipboard>
#include <QGuiApplication>
#include <QIODevice>
#include <QTableView>
#include <QWidget>

namespace annotation {

namespace details {

class AnnotationTableView : public QTableView {
public:
  explicit AnnotationTableView(QWidget *parent = nullptr);
  void keyPressEvent(QKeyEvent *event) override;

  void deleteSelectedRows();
  void copySelectedItems() const;
  void pasteItems();
};

} // namespace details

// Table view for inspecting and editing labels
class AnnotationView : public QWidget {
public:
  explicit AnnotationView(QWidget *parent = nullptr);
  void setModel(AnnotationModel *model);

private:
  AnnotationModel *m_model{};

  details::AnnotationTableView *m_tableView;
};

} // namespace annotation