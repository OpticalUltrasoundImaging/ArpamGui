#include "Annotation/GraphicsItemBase.hpp"
#include <QAction>
#include <QFont>
#include <Qt>
#include <utility>

namespace annotation {

GraphicsItemBase::GraphicsItemBase(Annotation annotation)
    :

      m_annotation(std::move(annotation)),
      m_textItem(new QGraphicsSimpleTextItem(this))

{
  setSelected(true);
  setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable |
           QGraphicsItem::ItemIsMovable);
  setAcceptHoverEvents(true);

  // By default the text is positioned at the first point in the polygon
  m_textItem->setPos(m_annotation.polygon[0]);
  m_textItem->setText(m_annotation.name);
  m_textItem->setBrush(m_annotation.color);
}

void GraphicsItemBase::updateScaleFactor(double scaleFactor) {
  prepareGeometryChange();
  QFont font;
  font.setPointSizeF(FONT_SIZE_BASE / scaleFactor);
  m_textItem->setFont(font);

  m_penWidth = PEN_WIDTH_BASE / scaleFactor;
}

auto GraphicsItemBase::pen() const -> QPen {
  QPen pen(color());
  pen.setWidthF(m_penWidth);
  pen.setCapStyle(Qt::PenCapStyle::RoundCap);
  pen.setStyle(m_penStyle);
  return pen;
}

void GraphicsItemBase::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {
  Q_UNUSED(event)
  m_inHover = true;
  update();
}

void GraphicsItemBase::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
  Q_UNUSED(event)
  m_inHover = false;
  update();
}

} // namespace annotation
