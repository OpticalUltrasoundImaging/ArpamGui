#pragma once
#include "Annotation/Annotation.hpp"
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
#include <cmath>
#include <functional>
#include <utility>

namespace annotation {

namespace details {

[[nodiscard]] inline auto getType(const Annotation &annotation) {
  return Annotation::typeToString(annotation.type());
}

[[nodiscard]] inline auto getName(const Annotation &annotation) {
  return annotation.name();
};

[[nodiscard]] inline auto getColor(const Annotation &annotation) {
  return annotation.color();
};

[[nodiscard]] inline auto setName(Annotation &annotation,
                                  const QVariant &value) {
  annotation.setName(value.toString());
};

struct ColumnMetaData {
  QString header;
  bool editable;
  std::function<QVariant(const Annotation &annotation)> getter;
  std::function<void(Annotation &annotation, const QVariant &value)> setter;
};

} // namespace details

class AnnotationModel : public QAbstractListModel {
  Q_OBJECT
public:
  enum AnnotationRoles { TypeRole = Qt::UserRole + 1, NameRole, ColorRole };

  using ColMeta = details::ColumnMetaData;

  // Column metadata
  inline static const std::array HEADER_DATA{
      ColMeta{"Type", false, details::getType},
      ColMeta{"Name", true, details::getName, details::setName},
      ColMeta{"Color", false, details::getColor}};

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
  void removeAnnotation(int row);

  [[nodiscard]] auto &front() { return m_annotations.front(); }
  [[nodiscard]] const auto &front() const { return m_annotations.front(); }
  [[nodiscard]] auto &back() { return m_annotations.back(); }
  [[nodiscard]] const auto &back() const { return m_annotations.back(); }

  [[nodiscard]] Annotation const &getAnnotation(int row) const {
    return m_annotations[row];
  }

  [[nodiscard]] auto size() { return m_annotations.size(); }

private:
  QList<Annotation> m_annotations;
};

#undef ANNO_GETTER
#undef ANNO_SETTER

} // namespace annotation