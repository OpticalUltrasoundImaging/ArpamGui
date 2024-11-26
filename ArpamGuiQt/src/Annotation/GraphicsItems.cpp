#include "GraphicsItems.hpp"
#include <kfr/cometa.hpp>
#include <qpainterpath.h>

namespace annotation {

void LineItem::setLine(const QLineF &line) {
  prepareGeometryChange();
  m_annotation.setLine(line);
  const auto textPos = std::max(line.p1(), line.p2(),
                                [](const QPointF &left, const QPointF &right) {
                                  return left.x() < right.x();
                                });
  setTextPos(textPos);
}

QRectF LineItem::boundingRect() const {
  auto rect = m_annotation.rect().normalized();
  constexpr double textWidth = 20; // Add space for text on the right side
  rect.setWidth(rect.width() + textWidth);
  return rect;
}

void LineItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                     QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)

  painter->setPen(pen());
  painter->drawLine(m_annotation.line());

  textItem()->setBrush(color());
}

void RectItem::setRect(const QRectF &rect) {
  prepareGeometryChange();
  m_annotation.setRect(rect);
  textItem()->setPos(rect.topLeft());
}

void RectItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                     QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)
  painter->setPen(pen());
  painter->drawRect(m_annotation.rect());

  textItem()->setBrush(color());
}

QPainterPath FanItem::getFanPainterPath() const {
  const auto rect = m_annotation.rect();
  const auto arc = m_annotation.arc();
  const auto center = rect.center();

  QPainterPath path;
  path.moveTo(center);
  path.arcTo(rect, arc.startAngle, arc.spanAngle);
  path.closeSubpath();

  return path;
}

QPainterPath FanItem::shape() const { return getFanPainterPath(); }

void FanItem::setRect(QRectF rect) {
  prepareGeometryChange();
  m_annotation.setRect(rect);
  needToUpdateTextPosition = true;
}

void FanItem::setArc(Arc arc) {
  prepareGeometryChange();
  m_annotation.setArc(arc);
  needToUpdateTextPosition = true;
}

void FanItem::setStartAngle(int angle) {
  prepareGeometryChange();
  m_annotation.setStartAngle(angle);
  needToUpdateTextPosition = true;
}

void FanItem::setSpanAngle(int angle) {
  prepareGeometryChange();
  m_annotation.setSpanAngle(angle);
  needToUpdateTextPosition = true;
}

void FanItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                    QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)

  if (needToUpdateTextPosition) {
    needToUpdateTextPosition = false;

    const auto startPoint = geometry::calcPosFromAngle(
        m_annotation.rect(), m_annotation.startAngle());
    textItem()->setPos(startPoint);
  }

  painter->setPen(pen());

  auto path = getFanPainterPath();
  painter->drawPath(path);

  textItem()->setBrush(color());
}

void PolygonItem::setPolygon(const QPolygonF &polygon) {
  prepareGeometryChange();
  m_annotation.polygon = polygon;
  textItem()->setPos(polygon.at(0));
}

void PolygonItem::paint(QPainter *painter,
                        const QStyleOptionGraphicsItem *option,
                        QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)
  painter->setPen(pen());

  painter->drawPolygon(m_annotation.polygon);

  textItem()->setBrush(color());
}

GraphicsItemBase *makeGraphicsItem(const Annotation &anno) {
  const auto type = anno.type;
  assert(type >= 0);
  assert(type < Annotation::Type::Size);

  switch (anno.type) {
  case Annotation::Type::Line:
    return new LineItem(anno);
  case Annotation::Type::Rect:
    return new RectItem(anno);
  case Annotation::Type::Fan:
    return new FanItem(anno);
  case Annotation::Type::Polygon:
    return new PolygonItem(anno);
  case Annotation::Type::Size:
    return nullptr;
  }
}

} // namespace annotation
