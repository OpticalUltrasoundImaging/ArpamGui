#pragma once

#include "CanvasAnnotationModel.hpp"

#include <QTableView>
#include <QWidget>

// Table view for inspecting and editing labels
class AnnotationView : public QWidget {
public:
  explicit AnnotationView(QWidget *parent = nullptr);
  void setModel(AnnotationModel *model) {
    m_model = model;
    m_tableView->setModel(m_model);
  }

private:
  AnnotationModel *m_model{};
  QTableView *m_tableView;
};
