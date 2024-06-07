#pragma once
#include <QAbstractListModel>
#include <QColor>
#include <QLineF>
#include <QList>
#include <QPolygonF>
#include <QRectF>
#include <QVariant>
#include <utility>

// Points are stored in a QPolygonF, which is just a QList<QPointF>
// https://doc.qt.io/qt-6/qpolygonf.html#details
class Annotation {
public:
  enum Type { Line, Rect, Polygon };

  /* Constructors */
  Annotation(Type type, const QList<QPointF> &points, const QColor &color);
  Annotation(const QLineF &line, const QColor &color);
  Annotation(const QRectF &rect, const QColor &color);

  [[nodiscard]] auto type() const { return m_type; }
  void setType(Type type) { m_type = type; }

  // For Line, the 2 points are {p1, p2}
  [[nodiscard]] auto line() const -> QLineF;

  // For Rect, the 2 points are {top_left, bottom_right}
  [[nodiscard]] auto rect() const -> QRectF;

  [[nodiscard]] auto polygon() const -> QPolygonF { return m_polygon; };
  void setPolygon(const QPolygonF &polygon) { m_polygon = polygon; }

  [[nodiscard]] auto color() const -> QColor { return m_color; }
  void setColor(QColor color) { m_color = color; }

private:
  Type m_type;
  QPolygonF m_polygon;
  QColor m_color;
};

class AnnotationModel : public QAbstractListModel {
  Q_OBJECT
public:
  enum AnnotationRoles { TypeRole = Qt::UserRole + 1, PolygonRole, ColorRole };

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
    case PolygonRole:
      return annotation.polygon();
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
      annotation.setType(static_cast<Annotation::Type>(value.toInt()));
      break;
    case PolygonRole:
      annotation.setPolygon(value.value<QPolygon>());
      break;
    case ColorRole:
      annotation.setColor(value.value<QColor>());
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

  [[nodiscard]] auto size() { return m_annotations.size(); }

private:
  QList<Annotation> m_annotations;
};
