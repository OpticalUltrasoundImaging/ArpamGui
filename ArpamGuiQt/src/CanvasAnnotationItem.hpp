/**
Subclasses of QGraphicsItem for annotation visualization
*/
#pragma once

#include "CanvasAnnotationModel.hpp"
#include "geometryUtils.hpp"
#include <QColor>
#include <QGraphicsItem>
#include <QGraphicsRectItem>
#include <QLineF>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QRectF>
#include <QTransform>
#include <QVariant>
#include <qnamespace.h>
#include <utility>

class AnnotationGraphicsItemBase : public QGraphicsItem {
public:
  inline const static QColor COLOR_HOVER = Qt::green;
  inline const static int PEN_WIDTH = 2;

  explicit AnnotationGraphicsItemBase(const QColor &color,
                                      QGraphicsItem *parent = nullptr)
      : QGraphicsItem(parent), m_color(color) {
    setSelected(true);
    setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable |
             QGraphicsItem::ItemIsMovable);
    setAcceptHoverEvents(true);
  }

  auto getPen() {
    if (m_inHover) [[unlikely]] {
      QPen pen(COLOR_HOVER);
      pen.setWidth(PEN_WIDTH);
      return pen;
    } else {
      QPen pen(m_color);
      pen.setWidth(PEN_WIDTH);
      return pen;
    }
  }

  [[nodiscard]] auto color() const { return m_color; }
  void setColor(const QColor &color) { m_color = color; }

  [[nodiscard]] auto name() const { return m_name; }
  void setName(QString name) { m_name = std::move(name); }

protected:
  void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override {
    Q_UNUSED(event)
    m_inHover = true;
    update();
  }

  void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override {
    Q_UNUSED(event)
    m_inHover = false;
    update();
  }

private:
  QColor m_color;
  QString m_name;
  bool m_inHover{false};
};

class LineItem : public AnnotationGraphicsItemBase {
public:
  LineItem(const QLineF &line, const QColor &color,
           QGraphicsItem *parent = nullptr)
      : AnnotationGraphicsItemBase(color, parent), m_line(line) {}

  void setLine(const QLineF &line) {
    if (line != m_line) {
      prepareGeometryChange();
      m_line = line;
    }
  }

  [[nodiscard]] QRectF boundingRect() const override {
    return QRectF(m_line.p1(), m_line.p2()).normalized();
  }

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override {
    Q_UNUSED(option)
    Q_UNUSED(widget)

    painter->setPen(getPen());

    painter->drawLine(m_line);
  }

private:
  QLineF m_line;
};

class RectItem : public AnnotationGraphicsItemBase {
public:
  RectItem(const QRectF &rect, const QColor &color,
           QGraphicsItem *parent = nullptr)
      : AnnotationGraphicsItemBase(color, parent), m_rect(rect) {}

  void setRect(const QRectF &rect) {
    if (rect != m_rect) {
      prepareGeometryChange();
      m_rect = rect;
    }
  }

  [[nodiscard]] QRectF boundingRect() const override { return m_rect; }

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override {
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setPen(getPen());
    painter->drawRect(m_rect);
  }

private:
  QRectF m_rect;
};

class FanItem : public AnnotationGraphicsItemBase {
public:
  FanItem(const QRectF &rect, Arc arc, const QColor &color,
          QGraphicsItem *parent = nullptr)
      : AnnotationGraphicsItemBase(color, parent), m_rect(rect), m_arc(arc) {}

  [[nodiscard]] QRectF boundingRect() const override {
    // TODO
    return m_rect;
  }

  [[nodiscard]] auto arc() const { return m_arc; }
  [[nodiscard]] auto startAngle() const { return m_arc.startAngle; }
  [[nodiscard]] auto spanAngle() const { return m_arc.spanAngle; }
  void setSpanAngle(int angle) {
    if (angle != m_arc.spanAngle) {
      prepareGeometryChange();
      m_arc.spanAngle = angle;
    }
  }

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override {
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setPen(getPen());

    // Draw arc
    painter->drawArc(m_rect, m_arc.startAngle, m_arc.spanAngle);

    // Draw the lines connecting the ends of the arc to the center
    const auto [startPoint, endPoint] =
        geometry::calcPosFromArc(m_rect, m_arc.startAngle, m_arc.spanAngle);

    painter->drawLine(m_rect.center(), startPoint);
    painter->drawLine(m_rect.center(), endPoint);
  }

private:
  // Pair of angles (each 0-360) that denote a fan shape center at the center of
  // the square image. The fan is drawn clockwise

  // Draws the arc defined by the given rectangle, startAngle and spanAngle.
  // The startAngle and spanAngle must be specified in 1/16th of a degree, i.e.
  // a full circle equals 5760 (16 * 360). Positive values for the angles mean
  // counter-clockwise while negative values mean the clockwise direction. Zero
  // degrees is at the 3 o'clock position.

  QRectF m_rect;
  Arc m_arc;
};
