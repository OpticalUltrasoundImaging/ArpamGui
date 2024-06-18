#include "GraphicsItems.hpp"
#include <qpainterpath.h>

namespace annotation {

void LineItem::updateAnno(const Annotation &anno) {
  prepareGeometryChange();
  setLine(anno.line());
  setColor(anno.color());
  setName(anno.name());
}

void LineItem::setLine(const QLineF &line) {
  if (line != m_line) {
    prepareGeometryChange();
    m_line = line;
  }
}

QRectF LineItem::boundingRect() const {
  return QRectF(m_line.p1(), m_line.p2()).normalized();
}

void LineItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                     QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)

  painter->setPen(getPen());

  painter->drawLine(m_line);

  painter->drawText(this->boundingRect(), name());
}

QPainterPath LineItem::shape() const {
  QPainterPath path;
  path.moveTo(m_line.p1());
  path.lineTo(m_line.p2());
  return path;
}

void RectItem::updateAnno(const Annotation &anno) {
  prepareGeometryChange();
  setRect(anno.rect());
  setColor(anno.color());
  setName(anno.name());
}

void RectItem::setRect(const QRectF &rect) {
  if (rect != m_rect) {
    prepareGeometryChange();
    m_rect = rect;
  }
}

void RectItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                     QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)
  painter->setPen(getPen());
  painter->drawRect(m_rect);

  painter->drawText(this->boundingRect(), name());
}

void FanItem::updateAnno(const Annotation &anno) {
  prepareGeometryChange();
  setArc(anno.arc());
  setColor(anno.color());
  setName(anno.name());
}

QPainterPath FanItem::getFanPainterPath() const {
  const auto startPoint = geometry::calcPosFromAngle(m_rect, m_arc.startAngle);

  QPainterPath path;
  path.moveTo(startPoint);
  path.arcTo(m_rect, m_arc.startAngle, m_arc.spanAngle);
  path.lineTo(m_rect.center());
  path.closeSubpath();

  return path;
}

QPainterPath FanItem::shape() const { return getFanPainterPath(); }

void FanItem::setArc(Arc arc) {
  prepareGeometryChange();
  m_arc = arc;
}

void FanItem::setStartAngle(int angle) {
  if (angle != m_arc.startAngle) {
    prepareGeometryChange();
    m_arc.startAngle = angle;
  }
}

void FanItem::setSpanAngle(int angle) {
  if (angle != m_arc.spanAngle) {
    prepareGeometryChange();
    m_arc.spanAngle = angle;
  }
}

void FanItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                    QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)
  painter->setPen(getPen());

  auto path = getFanPainterPath();

  painter->drawPath(path);

  const auto startPoint = geometry::calcPosFromAngle(m_rect, m_arc.startAngle);
  painter->drawText(startPoint, name());
}

void PolygonItem::updateAnno(const Annotation &anno) {
  prepareGeometryChange();
  setPolygon(anno.polygon());
  setColor(anno.color());
  setName(anno.name());
}

QRectF FanItem::boundingRect() const { return m_rect; }

void PolygonItem::setPolygon(const QPolygonF &polygon) {
  prepareGeometryChange();
  m_polygon = polygon;
}

void PolygonItem::paint(QPainter *painter,
                        const QStyleOptionGraphicsItem *option,
                        QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)
  painter->setPen(getPen());

  painter->drawPolygon(m_polygon);

  painter->drawText(this->boundingRect(), name());
}

} // namespace annotation