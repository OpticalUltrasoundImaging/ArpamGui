#include "CanvasAnnotationModel.hpp"
#include <cassert>
#include <qabstractitemmodel.h>
#include <qnamespace.h>

Annotation::Annotation(Type type, const QList<QPointF> &points,
                       const QColor &color)
    : m_type(type), m_polygon(points), m_color(color) {}

Annotation::Annotation(const QLineF &line, const QColor &color)
    : m_type(Line), m_polygon({line.p1(), line.p2()}), m_color(color) {}

Annotation::Annotation(const QRectF &rect, const QColor &color)
    : m_type(Rect), m_polygon({rect.topLeft(), rect.bottomRight()}),
      m_color(color) {}

auto Annotation::line() const -> QLineF {
  assert(m_polygon.size() == 2);
  return {m_polygon[0], m_polygon[1]};
};

auto Annotation::rect() const -> QRectF {
  assert(m_polygon.size() == 2);
  return {m_polygon[0], m_polygon[1]};
};

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
    return HEADER_DATA.at(section);
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
    if (index.column() == 0) {
      return annotation.color();
    }

    if (index.column() == 1) {
      return Annotation::typeToString(annotation.type());
    }

    if (index.column() == 2) {
      return annotation.name();
    }

    if (index.column() == 3) {
      return annotation.polygon();
    }

  case TypeRole:
    return annotation.type();
  case PolygonRole:
    return annotation.polygon();
  case ColorRole:
    return annotation.color();
  case NameRole:
    return annotation.name();
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
  switch (role) {
  case TypeRole:
    annotation.setType(static_cast<Annotation::Type>(value.toInt()));
    break;
  case PolygonRole:
    annotation.setPolygon(value.value<QPolygon>());
    break;
  case ColorRole:
    annotation.setColor(value.value<QColor>());
    break;
  case NameRole:
    annotation.setName(value.toString());
  default:
    return false;
  }

  emit dataChanged(index, index, {role});
  return true;
}

Qt::ItemFlags AnnotationModel::flags(const QModelIndex &index) const {
  if (!index.isValid()) {
    return Qt::NoItemFlags;
  }
  return QAbstractListModel::flags(index);
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