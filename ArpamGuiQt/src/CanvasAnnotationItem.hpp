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
#include <array>
#include <qnamespace.h>
#include <utility>
#include <vector>

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
  bool m_inHover{false};
};

class LineItem : public AnnotationGraphicsItemBase {
public:
  LineItem(const QLineF &line, const QColor &color,
           QGraphicsItem *parent = nullptr)
      : AnnotationGraphicsItemBase(color, parent), m_line(line) {
    setAcceptHoverEvents(true);
  }

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
      : AnnotationGraphicsItemBase(color, parent), m_rect(rect) {
    setAcceptHoverEvents(true);
  }

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
