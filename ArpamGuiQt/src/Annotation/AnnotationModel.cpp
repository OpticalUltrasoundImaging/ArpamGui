#include "Annotation/AnnotationModel.hpp"
#include <QAbstractItemModel>
#include <Qt>
#include <cassert>
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <utility>

namespace annotation {

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
    return annotation.type;
  case NameRole:
    return annotation.name;
  case ColorRole:
    return annotation.color;
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
    setDirty();
    emit dataChanged(index, index, {});
    return true;
  }
  if (role == NameRole) {
    annotation.name = value.toString();
    setDirty();
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
  setDirty();
  beginRemoveRows(parent, row, row + count - 1);
  m_annotations.remove(row, count);
  endRemoveRows();
  return true;
}

void AnnotationModel::addAnnotation(const Annotation &annotation) {
  setDirty();
  beginInsertRows(QModelIndex(), static_cast<int>(m_annotations.size()),
                  static_cast<int>(m_annotations.size()));
  m_annotations.append(annotation);
  endInsertRows();
}

void AnnotationModel::clear() {
  setDirty(false);
  if (size() > 0) {
    beginRemoveRows(QModelIndex(), 0,
                    static_cast<int>(m_annotations.size()) - 1);
    m_annotations.clear();
    endRemoveRows();
  }
}

rapidjson::Value AnnotationModel::serializeToJson(
    rapidjson::Document::AllocatorType &allocator) const {
  rapidjson::Value val(rapidjson::kObjectType);

  for (const auto &anno : m_annotations) {
    val.AddMember("annotations", anno.serializeToJson(allocator), allocator);
  }

  return val;
}

void AnnotationModel::deserializeFromJson(const rapidjson::Value &value) {
  clear();

  for (const auto &annoVal : value["annotations"].GetArray()) {
    addAnnotation(Annotation::deserializeFromJson(annoVal));
  }
}

void AnnotationModel::setAnnotations(QList<Annotation> annotations) {
  clear();

  if (!annotations.empty()) {
    beginInsertRows(QModelIndex{}, 0, static_cast<int>(annotations.size()) - 1);
    m_annotations = std::move(annotations);
    endInsertRows();
  }
}

} // namespace annotation
