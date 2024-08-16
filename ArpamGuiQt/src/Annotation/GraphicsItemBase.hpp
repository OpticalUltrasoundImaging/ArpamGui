#pragma once

#include "Annotation/Annotation.hpp"
#include <QAction>
#include <QColor>
#include <QGraphicsItem>
#include <QGraphicsSimpleTextItem>
#include <QPen>
#include <QPointF>
#include <Qt>

namespace annotation {

// Common base class for all QGraphicsItem representation of Annotation
class GraphicsItemBase : public QGraphicsItem {

public:
  inline const static QColor COLOR_HOVER = Qt::green;
  inline const static QColor COLOR_DEFAULT = Qt::white;
  inline const static double PEN_WIDTH_BASE = 2;
  inline const static double FONT_SIZE_BASE = 16.0;

  GraphicsItemBase() = delete;
  GraphicsItemBase(const GraphicsItemBase &) = delete;
  GraphicsItemBase(GraphicsItemBase &&) = delete;
  GraphicsItemBase &operator=(const GraphicsItemBase &) = delete;
  GraphicsItemBase &operator=(GraphicsItemBase &&) = delete;

  explicit GraphicsItemBase(Annotation annotation);
  ~GraphicsItemBase() override = default;

  [[nodiscard]] QRectF boundingRect() const override {
    return m_annotation.rect();
  }

  [[nodiscard]] auto color() const {
    if (m_inHover) [[unlikely]] {
      return COLOR_HOVER;
    }
    return m_annotation.color;
  }

  [[nodiscard]] auto annotation() const { return m_annotation; }
  void updateAnnotation(const Annotation &annotation) {
    prepareGeometryChange();
    m_annotation = annotation;
    m_textItem->setText(annotation.name);
    m_textItem->setBrush(annotation.color);
  }

  // Pen for drawing the shape outline
  [[nodiscard]] auto pen() const -> QPen;

  [[nodiscard]] auto penStyle() const { return m_penStyle; }
  void setPenStyle(Qt::PenStyle style) { m_penStyle = style; }

  // [[nodiscard]] auto text() const { return m_textGraphic->text(); }
  void setText(const QString &text) {
    m_textItem->setText(text);
    m_annotation.name = text;
  }
  [[nodiscard]] auto textItem() { return m_textItem; }
  void setTextPos(QPointF pos) { m_textItem->setPos(pos); }

  void updateScaleFactor(double scaleFactor);

protected:
  void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
  void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

  Annotation m_annotation;

private:
  QGraphicsSimpleTextItem *m_textItem{}; // Child item
  double m_penWidth{PEN_WIDTH_BASE};
  Qt::PenStyle m_penStyle{Qt::PenStyle::SolidLine};
  bool m_inHover{false};
};

} // namespace annotation
