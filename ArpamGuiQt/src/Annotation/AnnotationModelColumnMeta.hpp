#pragma once

#include "Annotation/Annotation.hpp"
#include <QString>
#include <QVariant>
#include <functional>

namespace annotation {

/**
 * Column metadata for the table view columns
 */
struct ColumnMetaData {
  QString header;
  bool editable;
  std::function<QVariant(const Annotation &annotation)> getter;
  std::function<void(Annotation &annotation, const QVariant &value)> setter;
};

} // namespace annotation