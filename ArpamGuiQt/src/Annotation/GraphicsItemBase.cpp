#include "Annotation/GraphicsItemBase.hpp"

namespace annotation {

GraphicsItemBase::GraphicsItemBase() : m_color(COLOR_DEFAULT) {}

GraphicsItemBase::GraphicsItemBase(const Annotation &annotation,
                                   QGraphicsItem *parent)
    : QGraphicsItem(parent), m_color(annotation.color()) {
  setSelected(true);
  setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable |
           QGraphicsItem::ItemIsMovable);
  setAcceptHoverEvents(true);
}

GraphicsItemBase::GraphicsItemBase(const QColor &color, QGraphicsItem *parent)
    : QGraphicsItem(parent), m_color(color) {
  setSelected(true);
  setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable |
           QGraphicsItem::ItemIsMovable);
  setAcceptHoverEvents(true);
}

GraphicsItemBase::~GraphicsItemBase() = default;

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