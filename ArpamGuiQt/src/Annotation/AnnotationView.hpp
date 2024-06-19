#pragma once

#include "Annotation/AnnotationModel.hpp"
#include <QTableView>
#include <QWidget>

namespace annotation {

namespace details {

class AnnotationTableView : public QTableView {
public:
  explicit AnnotationTableView(QWidget *parent = nullptr);
  void keyPressEvent(QKeyEvent *event) override;
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