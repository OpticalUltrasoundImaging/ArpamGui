#pragma once
#include "geometryUtils.hpp"
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

/**
Pair of angles (each 0-360) that denote a fan shape center at the center of
the square image.

Draws the arc defined by the given rectangle, startAngle and spanAngle.
The startAngle and spanAngle are in degrees
Positive values for the angles mean counter-clockwise
while negative values mean the clockwise direction.
Zero degrees is at the 3 o'clock position.
*/
struct Arc {
  double startAngle;
  double spanAngle;

  [[nodiscard]] inline auto startAngleRadians() const {
    return geometry::deg2rad(startAngle);
  }

  [[nodiscard]] inline auto spanAngleRadians() const {
    return geometry::deg2rad(spanAngle);
  }
};

// Points are stored in a QPolygonF, which is just a QList<QPointF>
// https://doc.qt.io/qt-6/qpolygonf.html#details
class Annotation {
public:
  enum Type { Line, Rect, Fan, Polygon, Size };

  /* Constructors */
  Annotation(Type type, const QList<QPointF> &points, const QColor &color,
             QString name = {});
  Annotation(const QLineF &line, const QColor &color, QString name = {});
  Annotation(const QRectF &rect, const QColor &color, QString name = {});
  /* For arc, the Polygon contains 3 points that store
   * {rect.topLeft, rect.bottomRight, {startAngle, spanAngle}}
   */
  Annotation(const Arc &arc, const QRectF &rect, const QColor &color,
             QString name = {});

  /* Copy constructor */
  Annotation(const Annotation &other) = default;
  /* Copy assignment */
  Annotation &operator=(const Annotation &other) = default;
  /* Move constructor */
  Annotation(Annotation &&) = default;
  /* Move assignment */
  Annotation &operator=(Annotation &&) = default;

  ~Annotation() = default;

  [[nodiscard]] auto type() const { return m_type; }
  void setType(Type type) { m_type = type; }

  // For Line, the 2 points are {p1, p2}
  [[nodiscard]] auto line() const -> QLineF {
    assert(m_polygon.size() == 2);
    return {m_polygon[0], m_polygon[1]};
  };

  // For Rect, the 2 points are {top_left, bottom_right}
  [[nodiscard]] auto rect() const -> QRectF {
    assert(m_polygon.size() >= 2);
    return {m_polygon[0], m_polygon[1]};
  };

  // For arc, the 3rd point stores the startAngle (x) and spanAngle (y)
  [[nodiscard]] auto arc() const -> Arc {
    assert(m_polygon.size() == 3);
    const auto pt = m_polygon[2];
    return Arc{pt.x(), pt.y()};
  }

  [[nodiscard]] auto polygon() const -> QPolygonF { return m_polygon; };
  void setPolygon(const QPolygonF &polygon) { m_polygon = polygon; }

  [[nodiscard]] auto color() const -> QColor { return m_color; }
  void setColor(QColor color) { m_color = color; }

  [[nodiscard]] auto name() const { return m_name; }
  void setName(QString name) { m_name = std::move(name); }

  static QString typeToString(Type type) {
    switch (type) {
    case Line:
      return "Line";
    case Rect:
      return "Rect";
    case Fan:
      return "Fan";
    case Polygon:
      return "Polygon";
    }
  }

private:
  Type m_type;
  QPolygonF m_polygon;
  QColor m_color;
  QString m_name;
};

namespace annoGetSet {

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

} // namespace annoGetSet
class AnnotationModel : public QAbstractListModel {
  Q_OBJECT
public:
  enum AnnotationRoles { TypeRole = Qt::UserRole + 1, NameRole, ColorRole };

  // Column metadata
  struct ColumnMetaData {
    QString header;
    bool editable;
    std::function<QVariant(const Annotation &annotation)> getter;
    std::function<void(Annotation &annotation, const QVariant &value)> setter;
  };
  inline static const std::array HEADER_DATA{
      ColumnMetaData{"Type", false, annoGetSet::getType},
      ColumnMetaData{"Name", true, annoGetSet::getName, annoGetSet::setName},
      ColumnMetaData{
          "Color",
          false,
          annoGetSet::getColor,
      }};

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