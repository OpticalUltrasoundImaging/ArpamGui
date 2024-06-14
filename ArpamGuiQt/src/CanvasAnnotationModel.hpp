#pragma once
#include <QAbstractListModel>
#include <QColor>
#include <QLineF>
#include <QList>
#include <QPolygonF>
#include <QRectF>
#include <QString>
#include <QVariant>
#include <Qt>
#include <array>
#include <utility>

struct Arc {
  // Pair of angles (each 0-360) that denote a fan shape center at the center of
  // the square image. The fan is drawn clockwise

  // Draws the arc defined by the given rectangle, startAngle and spanAngle.
  // The startAngle and spanAngle must be specified in 1/16th of a degree, i.e.
  // a full circle equals 5760 (16 * 360). Positive values for the angles mean
  // counter-clockwise while negative values mean the clockwise direction. Zero
  // degrees is at the 3 o'clock position.
  int startAngle;
  int spanAngle;
};

// Points are stored in a QPolygonF, which is just a QList<QPointF>
// https://doc.qt.io/qt-6/qpolygonf.html#details
class Annotation {
public:
  enum Type { Line, Rect, Fan, Polygon };

  /* Constructors */
  Annotation(Type type, const QList<QPointF> &points, const QColor &color);
  Annotation(const QLineF &line, const QColor &color);
  Annotation(const QRectF &rect, const QColor &color);
  Annotation(const Arc &arc, const QColor &color);

  [[nodiscard]] auto type() const { return m_type; }
  void setType(Type type) { m_type = type; }

  // For Line, the 2 points are {p1, p2}
  [[nodiscard]] auto line() const -> QLineF;

  // For Rect, the 2 points are {top_left, bottom_right}
  [[nodiscard]] auto rect() const -> QRectF;

  // For arc, the 1 points stores the startAngle (x) and spanAngle (y)
  [[nodiscard]] auto arc() const -> Arc;

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

class AnnotationModel : public QAbstractListModel {
  Q_OBJECT
public:
  enum AnnotationRoles {
    TypeRole = Qt::UserRole + 1,
    PolygonRole,
    ColorRole,
    NameRole
  };
  const inline static std::array<QString, 4> HEADER_DATA{"Color", "Type",
                                                         "Name", "Points"};

  [[nodiscard]] int rowCount(const QModelIndex &parent) const override;
  [[nodiscard]] int columnCount(const QModelIndex &parent) const override;

  [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation,
                                    int role) const override;

  [[nodiscard]] QVariant data(const QModelIndex &index,
                              int role) const override;

  bool setData(const QModelIndex &index, const QVariant &value,
               int role) override;

  [[nodiscard]] Qt::ItemFlags flags(const QModelIndex &index) const override;

  void addAnnotation(const Annotation &annotation);

  void removeAnnotation(int row);

  [[nodiscard]] Annotation const &getAnnotation(int row) const {
    return m_annotations[row];
  }

  [[nodiscard]] auto size() { return m_annotations.size(); }

private:
  QList<Annotation> m_annotations;
};
