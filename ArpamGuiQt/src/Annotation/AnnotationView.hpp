#pragma once

#include "Annotation/AnnotationModel.hpp"
#include <QTableView>
#include <QWidget>

namespace annotation {

// Table view for inspecting and editing labels
class AnnotationView : public QWidget {
public:
  explicit AnnotationView(QWidget *parent = nullptr);
  void setModel(AnnotationModel *model);

private:
  AnnotationModel *m_model{};

  QTableView *m_tableView;
};

} // namespace annotation