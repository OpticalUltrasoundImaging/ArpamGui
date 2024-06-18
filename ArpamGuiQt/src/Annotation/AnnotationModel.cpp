#include "Annotation/AnnotationModel.hpp"
#include <QAbstractItemModel>
#include <Qt>
#include <cassert>
#include <utility>

namespace annotation {

Annotation::Annotation(Type type, const QList<QPointF> &points,
                       const QColor &color, QString name)
    : m_type(type), m_polygon(points), m_color(color), m_name(std::move(name)) {
}

Annotation::Annotation(const QLineF &line, const QColor &color, QString name)
    : m_type(Line), m_polygon({line.p1(), line.p2()}), m_color(color),
      m_name(std::move(name)) {}

Annotation::Annotation(const QRectF &rect, const QColor &color, QString name)
    : m_type(Rect), m_polygon({rect.topLeft(), rect.bottomRight()}),
      m_color(color), m_name(std::move(name)) {}

Annotation::Annotation(const Arc &arc, const QRectF &rect, const QColor &color,
                       QString name)
    : m_type(Fan), m_polygon({rect.topLeft(), rect.bottomRight(),
                              QPointF{arc.startAngle, arc.spanAngle}}),
      m_color(color), m_name(std::move(name)) {}

/*******/

[[nodiscard]] int AnnotationModel::rowCount(const QModelIndex &parent) const {
  Q_UNUSED(parent);
  return static_cast<int>(m_annotations.size());
}

[[nodiscard]] int
AnnotationModel::columnCount(const QModelIndex &parent) const {
  return HEADER_DATA.size();
}

[[nodiscard]] QVariant AnnotationModel::headerData(int section,
                                                   Qt::Orientation orientation,
                                                   int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
    assert(section < HEADER_DATA.size());
    return HEADER_DATA.at(section).header;
  }
  return QAbstractListModel::headerData(section, orientation, role);
}

[[nodiscard]] QVariant AnnotationModel::data(const QModelIndex &index,
                                             int role) const {
  if (!index.isValid() || index.row() >= m_annotations.size()) {
    return {};
  }

  const auto &annotation = m_annotations[index.row()];

  switch (role) {
  case Qt::DisplayRole:
    assert(index.column() < HEADER_DATA.size());
    return HEADER_DATA.at(index.column()).getter(annotation);

  case TypeRole:
    return annotation.type();
  case NameRole:
    return annotation.name();
  case ColorRole:
    return annotation.color();
  default:
    return {};
  }
}

bool AnnotationModel::setData(const QModelIndex &index, const QVariant &value,
                              int role) {
  if (!index.isValid() || index.row() >= m_annotations.size()) {
    return false;
  }

  auto &annotation = m_annotations[index.row()];
  if (role == Qt::EditRole) {
    HEADER_DATA.at(index.column()).setter(annotation, value);
    emit dataChanged(index, index, {});
    return true;
  }

  return false;
}

Qt::ItemFlags AnnotationModel::flags(const QModelIndex &index) const {
  if (!index.isValid()) {
    return Qt::NoItemFlags;
  }
  auto flags = QAbstractListModel::flags(index);
  if (HEADER_DATA.at(index.column()).editable) {
    flags |= Qt::ItemIsEditable;
  }

  return flags;
}

bool AnnotationModel::removeRows(int row, int count,
                                 const QModelIndex &parent) {
  if (row < 0 || count <= 0 || (row + count) > m_annotations.size()) {
    return false;
  }

  beginRemoveRows(parent, row, row + count - 1);
  m_annotations.remove(row, count);
  endRemoveRows();
  return true;
}

void AnnotationModel::addAnnotation(const Annotation &annotation) {
  beginInsertRows(QModelIndex(), m_annotations.size(), m_annotations.size());
  m_annotations.append(annotation);
  endInsertRows();
}

void AnnotationModel::removeAnnotation(int row) {
  beginRemoveRows(QModelIndex(), row, row);
  m_annotations.removeAt(row);
  endRemoveRows();
}

} // namespace annotation