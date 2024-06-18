#include "Annotation/GraphicsItemBase.hpp"

namespace annotation {

GraphicsItemBase::GraphicsItemBase()
    : m_color(COLOR_DEFAULT), m_textGraphic(new QGraphicsSimpleTextItem(this)) {
}

GraphicsItemBase::GraphicsItemBase(const Annotation &annotation,
                                   QGraphicsItem *parent)
    : QGraphicsItem(parent), m_textGraphic(new QGraphicsSimpleTextItem(this)),
      m_color(annotation.color()) {
  setSelected(true);
  setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable |
           QGraphicsItem::ItemIsMovable);
  setAcceptHoverEvents(true);

  QFont font;
  font.setPointSizeF(FONT_SIZE_BASE);
}

GraphicsItemBase::GraphicsItemBase(const QColor &color, QGraphicsItem *parent)
    : QGraphicsItem(parent), m_textGraphic(new QGraphicsSimpleTextItem(this)),
      m_color(color) {
  setSelected(true);
  setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable |
           QGraphicsItem::ItemIsMovable);
  setAcceptHoverEvents(true);
}

GraphicsItemBase::~GraphicsItemBase() = default;

void GraphicsItemBase::updateScaleFactor(double scaleFactor) {
  prepareGeometryChange();
  QFont font;
  font.setPointSizeF(FONT_SIZE_BASE / scaleFactor);
  textItem()->setFont(font);
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