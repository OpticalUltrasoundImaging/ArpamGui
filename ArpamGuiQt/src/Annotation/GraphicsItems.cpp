#include "GraphicsItems.hpp"
#include <qpainterpath.h>

namespace annotation {

void LineItem::updateAnno(const Annotation &anno) {
  prepareGeometryChange();
  const auto line = anno.line();
  setLine(line);
  setColor(anno.color());
  setText(anno.name());
}

void LineItem::setLine(const QLineF &line) {
  if (line != m_line) {
    prepareGeometryChange();
    m_line = line;

    auto pos = std::max(line.p1(), line.p2(),
                        [](const QPointF &left, const QPointF &right) {
                          return left.x() < right.x();
                        });
    textItem()->setPos(pos);
  }
}

QRectF LineItem::boundingRect() const {
  auto rect = QRectF(m_line.p1(), m_line.p2()).normalized();
  constexpr double textWidth = 20; // Add space for text on the right side
  rect.setWidth(rect.width() + textWidth);
  return rect;
}

void LineItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                     QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)

  painter->setPen(pen());
  painter->drawLine(m_line);

  textItem()->setBrush(color());
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
  setText(anno.name());

  textItem()->setBrush(color());
}

void RectItem::setRect(const QRectF &rect) {
  if (rect != m_rect) {
    prepareGeometryChange();
    m_rect = rect;

    textItem()->setPos(rect.topLeft());
  }
}

void RectItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                     QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)
  painter->setPen(pen());
  painter->drawRect(m_rect);

  textItem()->setBrush(color());
}

void FanItem::updateAnno(const Annotation &anno) {
  prepareGeometryChange();
  setArc(anno.arc());
  setRect(anno.rect());
  setColor(anno.color());
  setText(anno.name());
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
  needToUpdateTextPosition = true;
}

void FanItem::setRect(QRectF rect) {
  prepareGeometryChange();
  m_rect = rect;
  needToUpdateTextPosition = true;
}

void FanItem::setStartAngle(int angle) {
  if (angle != m_arc.startAngle) {
    prepareGeometryChange();
    m_arc.startAngle = angle;
    needToUpdateTextPosition = true;
  }
}

void FanItem::setSpanAngle(int angle) {
  if (angle != m_arc.spanAngle) {
    prepareGeometryChange();
    m_arc.spanAngle = angle;
    needToUpdateTextPosition = true;
  }
}

void FanItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                    QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)

  const auto startPoint = geometry::calcPosFromAngle(m_rect, m_arc.startAngle);

  if (needToUpdateTextPosition) {
    needToUpdateTextPosition = false;

    textItem()->setPos(startPoint);
  }

  painter->setPen(pen());

  auto path = getFanPainterPath();
  painter->drawPath(path);

  textItem()->setBrush(color());
}

void PolygonItem::updateAnno(const Annotation &anno) {
  prepareGeometryChange();
  setPolygon(anno.polygon());
  setColor(anno.color());
  setText(anno.name());
}

QRectF FanItem::boundingRect() const { return m_rect; }

void PolygonItem::setPolygon(const QPolygonF &polygon) {
  prepareGeometryChange();
  m_polygon = polygon;
  textItem()->setPos(polygon.at(0));
}

void PolygonItem::paint(QPainter *painter,
                        const QStyleOptionGraphicsItem *option,
                        QWidget *widget) {
  Q_UNUSED(option)
  Q_UNUSED(widget)
  painter->setPen(pen());

  painter->drawPolygon(m_polygon);

  textItem()->setBrush(color());
}

namespace details {

// Must keep order consistent with Annotation::Type
const static std::array makeFuncs = {
    makeItem<Annotation::Type::Line>, makeItem<Annotation::Type::Rect>,
    makeItem<Annotation::Type::Fan>, makeItem<Annotation::Type::Polygon>};
static_assert(makeFuncs.size() == Annotation::Type::Size);

} // namespace details

GraphicsItemBase *makeGraphicsItem(const Annotation &anno) {
  const auto type = anno.type();
  assert(type >= 0);
  assert(type < Annotation::Type::Size);
  return details::makeFuncs.at(type)(anno);
}

} // namespace annotation
