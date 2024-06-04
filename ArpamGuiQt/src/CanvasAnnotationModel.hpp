#pragma once
#include <QAbstractListModel>
#include <QColor>
#include <QList>
#include <QRectF>
#include <QVariant>

class Annotation {
public:
  enum Type { Line, Box };
  Annotation(Type type, const QRectF &rect, const QColor &color)
      : m_type(type), m_rect(rect), m_color(color) {}

  [[nodiscard]] auto type() const { return m_type; }
  [[nodiscard]] auto rect() const { return m_rect; }
  [[nodiscard]] auto color() const { return m_color; }

private:
  Type m_type;
  QRectF m_rect;
  QColor m_color;
};

class AnnotationModel : public QAbstractListModel {
  Q_OBJECT
public:
  enum AnnotationRoles { TypeRole = Qt::UserRole + 1, RectRole, ColorRole };

  [[nodiscard]] int rowCount(const QModelIndex &parent) const override {
    Q_UNUSED(parent);
    return static_cast<int>(m_annotations.size());
  }

  [[nodiscard]] QVariant data(const QModelIndex &index,
                              int role) const override {
    if (!index.isValid() || index.row() >= m_annotations.size()) {
      return {};
    }

    const Annotation &annotation = m_annotations[index.row()];

    switch (role) {
    case TypeRole:
      return annotation.type();
    case RectRole:
      return annotation.rect();
    case ColorRole:
      return annotation.color();
    default:
      return {};
    }
  }

  bool setData(const QModelIndex &index, const QVariant &value,
               int role) override {
    if (!index.isValid() || index.row() >= m_annotations.size()) {
      return false;
    }

    auto &annotation = m_annotations[index.row()];
    switch (role) {
    case TypeRole:
      annotation = Annotation(static_cast<Annotation::Type>(value.toInt()),
                              annotation.rect(), annotation.color());
      break;
    case RectRole:
      annotation =
          Annotation(annotation.type(), value.toRectF(), annotation.color());
      break;
    case ColorRole:
      annotation = Annotation(annotation.type(), annotation.rect(),
                              value.value<QColor>());
      break;
    default:
      return false;
    }

    emit dataChanged(index, index, {role});
    return true;
  }

  [[nodiscard]] Qt::ItemFlags flags(const QModelIndex &index) const override {
    if (!index.isValid()) {
      return Qt::NoItemFlags;
    }

    return Qt::ItemIsEditable | QAbstractListModel::flags(index);
  }

  void addAnnotation(const Annotation &annotation) {
    beginInsertRows(QModelIndex(), m_annotations.size(), m_annotations.size());
    m_annotations.append(annotation);
    endInsertRows();
  }

  void removeAnnotation(int row) {
    beginRemoveRows(QModelIndex(), row, row);
    m_annotations.removeAt(row);
    endRemoveRows();
  }

  [[nodiscard]] Annotation const &getAnnotation(int row) const {
    return m_annotations[row];
  }

private:
  QList<Annotation> m_annotations;
};
