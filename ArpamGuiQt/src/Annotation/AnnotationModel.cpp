#include "Annotation/AnnotationModel.hpp"
#include <QAbstractItemModel>
#include <QByteArray>
#include <QMimeData>
#include <Qt>
#include <QtLogging>
#include <cassert>
#include <rapidjson/document.h>
#include <rapidjson/error/error.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <set>
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

// NOLINTNEXTLINE(*-static)
QStringList AnnotationModel::mimeType() const {
  QStringList types;
  types << "application/json";
  return types;
}

QMimeData *AnnotationModel::mimeData(const QModelIndexList &indexes) const {
  // JSON impl

  // Find unique rows in selected annotations
  std::set<int> uniqueRows;
  for (const auto &index : indexes) {
    if (index.isValid()) {
      uniqueRows.insert(index.row());
    }
  }

  // Serialize annotations to JSON doc
  rapidjson::Document doc;
  doc.SetArray();
  auto &alloc = doc.GetAllocator();
  for (const auto row : uniqueRows) {
    doc.PushBack(m_annotations[row].serializeToJson(alloc), alloc);
  }

  // Serialize JSON to string buffer and QByteArray
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  auto *mimeData = new QMimeData;
  // Implicitly construct QByteArray from const char *
  mimeData->setData("application/json", buffer.GetString());
  return mimeData;
}

bool AnnotationModel::canDropMimeData(const QMimeData *data,
                                      Qt::DropAction action, int row,
                                      int column,
                                      const QModelIndex &parent) const {
  return data->hasFormat("application/json");
}

bool AnnotationModel::dropMimeData(const QMimeData *data, Qt::DropAction action,
                                   int row, int column,
                                   const QModelIndex &parent) {
  // JSON impl
  if (!data->hasFormat("application/json")) {
    return false;
  }

  const QByteArray encodedData = data->data("application/json");

  rapidjson::Document doc;
  rapidjson::ParseResult parseResult = doc.Parse(encodedData.constData());
  if ((parseResult != nullptr) && doc.IsArray()) {
    // parse array
    for (const auto &item : doc.GetArray()) {
      const auto anno = Annotation::deserializeFromJson(item);
      addAnnotation(anno);
    }
    return true;
  }

  return false;
}

} // namespace annotation