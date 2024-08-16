#pragma once

#include "Annotation/Annotation.hpp"
#include "Annotation/GraphicsItemBase.hpp"
#include <QGraphicsSimpleTextItem>
#include <QLineF>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPolygon>
#include <QRectF>

namespace annotation {

class LineItem : public GraphicsItemBase {
public:
  explicit LineItem(const Annotation &anno) : GraphicsItemBase(anno) {
    textItem()->setPos(m_annotation.polygon[1]);
  }

  void setLine(const QLineF &line);

  [[nodiscard]] QRectF boundingRect() const override;

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override;
};

class RectItem : public GraphicsItemBase {
public:
  explicit RectItem(const Annotation &anno) : GraphicsItemBase(anno) {}

  void setRect(const QRectF &rect);

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override;
};

// Pair of angles (each 0-360) that denote a fan shape center at the center of
// the square image. The fan is drawn clockwise

// Draws the arc defined by the given rectangle, startAngle and spanAngle.
// The startAngle and spanAngle must be specified in 1/16th of a degree, i.e.
// a full circle equals 5760 (16 * 360). Positive values for the angles mean
// counter-clockwise while negative values mean the clockwise direction. Zero
// degrees is at the 3 o'clock position.
class FanItem : public GraphicsItemBase {
public:
  explicit FanItem(const Annotation &anno) : GraphicsItemBase(anno) {}

  [[nodiscard]] QPainterPath getFanPainterPath() const;

  [[nodiscard]] QPainterPath shape() const override;

  [[nodiscard]] auto rect() const { return m_annotation.rect(); }
  void setRect(QRectF rect);

  [[nodiscard]] auto arc() const { return m_annotation.arc(); }
  void setArc(Arc arc);

  [[nodiscard]] auto startAngle() const { return m_annotation.startAngle(); }
  void setStartAngle(int angle);

  [[nodiscard]] auto spanAngle() const { return m_annotation.spanAngle(); }
  void setSpanAngle(int angle);

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override;

private:
  bool needToUpdateTextPosition{true};
};

class PolygonItem : public GraphicsItemBase {
public:
  explicit PolygonItem(const Annotation &anno) : GraphicsItemBase(anno) {}

  void setPolygon(const QPolygonF &polygon);

  [[nodiscard]] QRectF boundingRect() const override {
    return m_annotation.polygon.boundingRect();
  }

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override;
};

// Create an instance of the correct graphics item for the given annotation
GraphicsItemBase *makeGraphicsItem(const Annotation &anno);

} // namespace annotation
