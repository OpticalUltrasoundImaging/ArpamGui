#include "Annotation/GraphicsItemBase.hpp"
#include <QFont>
#include <Qt>

namespace annotation {

GraphicsItemBase::GraphicsItemBase()
    : m_color(COLOR_DEFAULT), m_textGraphic(new QGraphicsSimpleTextItem(this)) {
}

GraphicsItemBase::GraphicsItemBase(const Annotation &annotation,
                                   QGraphicsItem *parent)
    : QGraphicsItem(parent), m_textGraphic(new QGraphicsSimpleTextItem(this)),
      m_color(annotation.color) {
  setSelected(true);
  setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable |
           QGraphicsItem::ItemIsMovable);
  setAcceptHoverEvents(true);
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
