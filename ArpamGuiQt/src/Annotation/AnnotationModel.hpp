#pragma once

#include "Annotation/Annotation.hpp"
#include "Annotation/AnnotationModelColumnMeta.hpp"
#include <QAbstractListModel>
#include <QColor>
#include <QColorDialog>
#include <QLineF>
#include <QList>
#include <QPolygonF>
#include <QRectF>
#include <QString>
#include <QVariant>
#include <Qt>
#include <array>
#include <rapidjson/document.h>

namespace annotation {

class AnnotationModel : public QAbstractListModel {
  Q_OBJECT
public:
  enum AnnotationRoles { TypeRole = Qt::UserRole + 1, NameRole, ColorRole };

  [[nodiscard]] bool dirty() const { return m_dirty; }
  void setDirty(bool state = true) { m_dirty = state; }

  [[nodiscard]] auto &annotations() const { return m_annotations; }
  void setAnnotations(QList<Annotation> annotations);

  using ColMeta = ColumnMetaData;

  // Column metadata
  inline static const std::array HEADER_DATA{
      ColMeta{"Type", false,
              [](const Annotation &anno) {
                return Annotation::typeToString(anno.type);
              }},
      ColMeta{"Name", true, [](const Annotation &anno) { return anno.name; },
              [](Annotation &anno, const QVariant &val) {
                anno.name = val.toString();
              }},
      ColMeta{"Color", false,
              [](const Annotation &anno) { return anno.color; }}};

  [[nodiscard]] int rowCount(const QModelIndex &parent) const override;
  [[nodiscard]] int columnCount(const QModelIndex &parent) const override;

  [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation,
                                    int role) const override;

  [[nodiscard]] QVariant data(const QModelIndex &index,
                              int role) const override;

  bool setData(const QModelIndex &index, const QVariant &value,
               int role) override;

  [[nodiscard]] Qt::ItemFlags flags(const QModelIndex &index) const override;

  bool removeRows(int row, int count, const QModelIndex &parent) override;

  void addAnnotation(const Annotation &annotation);

  // It's equivalent to call removeRow(int)
  // void removeAnnotation(int row);

  [[nodiscard]] auto &front() { return m_annotations.front(); }
  [[nodiscard]] const auto &front() const { return m_annotations.front(); }
  [[nodiscard]] auto &back() { return m_annotations.back(); }
  [[nodiscard]] const auto &back() const { return m_annotations.back(); }

  [[nodiscard]] Annotation const &getAnnotation(int row) const {
    return m_annotations[row];
  }

  [[nodiscard]] Annotation &at(int row) { return m_annotations[row]; }
  [[nodiscard]] const Annotation &at(int row) const {
    return m_annotations[row];
  }

  [[nodiscard]] auto size() { return m_annotations.size(); }

  void clear();

  [[nodiscard]] rapidjson::Value
  serializeToJson(rapidjson::Document::AllocatorType &allocator) const;
  void deserializeFromJson(const rapidjson::Value &value);

private:
  QList<Annotation> m_annotations;
  bool m_dirty{false};
};

#undef ANNO_GETTER
#undef ANNO_SETTER

} // namespace annotation
