#pragma once

#include "Annotation/AnnotationModel.hpp"

#include <QAbstractItemView>
#include <QPainter>
#include <QString>
#include <QTableView>
#include <QVariant>
#include <QWidget>

namespace annotation {

// Table view for inspecting and editing labels
class AnnotationView : public QWidget {
public:
  explicit AnnotationView(QWidget *parent = nullptr);
  void setModel(AnnotationModel *model) {
    m_model = model;
    m_tableView->setModel(m_model);

    m_tableView->setSelectionBehavior(QAbstractItemView::SelectRows);
  }

private:
  AnnotationModel *m_model{};
  QTableView *m_tableView;
};

#include <QPainter>
#include <QStyledItemDelegate>

class AnnotationDelegate : public QStyledItemDelegate {
public:
  using QStyledItemDelegate::QStyledItemDelegate;

  void paint(QPainter *painter, const QStyleOptionViewItem &option,
             const QModelIndex &index) const override {
    if (index.isValid()) {
      painter->save();

      QVariant type = index.data(AnnotationModel::TypeRole);
      QVariant color = index.data(AnnotationModel::ColorRole);
      QVariant name = index.data(AnnotationModel::NameRole);

      if (color.isValid() && color.canConvert<QColor>()) {
        painter->setBrush(QBrush(color.value<QColor>()));
        painter->drawRect(
            option.rect.adjusted(0, 0, -option.rect.width() * 2 / 3, 0));
      }

      if (type.isValid() && type.canConvert<Annotation::Type>()) {
        const auto typeStr =
            Annotation::typeToString(type.value<Annotation::Type>());
        painter->drawText(
            option.rect.adjusted(option.rect.width() / 3, 0, 0, 0),
            Qt::AlignLeft, typeStr);
      }

      if (name.isValid() && name.canConvert<QString>()) {
        const auto nameStr = name.toString();
        painter->drawText(option.rect.adjusted(option.rect.width() / 3,
                                               option.rect.height() / 2, 0, 0),
                          Qt::AlignLeft, nameStr);
      }

      painter->restore();
    } else {
      QStyledItemDelegate::paint(painter, option, index);
    }
  }
};

} // namespace annotation