#pragma once

#include "Annotation/AnnotationModel.hpp"
#include "geometryUtils.hpp"
#include <QColor>
#include <QGraphicsItem>
#include <QGraphicsSimpleTextItem>
#include <QPen>
#include <QPointF>
#include <Qt>
#include <utility>

namespace annotation {

class GraphicsItemBase : public QGraphicsItem {
public:
  inline const static int PEN_WIDTH = 2;
  inline const static QColor COLOR_HOVER = Qt::green;
  inline const static QColor COLOR_DEFAULT = Qt::white;
  inline const static double FONT_SIZE_BASE = 16.0;

  GraphicsItemBase();
  GraphicsItemBase(const GraphicsItemBase &) = delete;
  GraphicsItemBase(GraphicsItemBase &&) = delete;
  GraphicsItemBase &operator=(const GraphicsItemBase &) = delete;
  GraphicsItemBase &operator=(GraphicsItemBase &&) = delete;

  explicit GraphicsItemBase(const Annotation &annotation,
                            QGraphicsItem *parent = nullptr);
  explicit GraphicsItemBase(const QColor &color,
                            QGraphicsItem *parent = nullptr);
  virtual ~GraphicsItemBase() override;

  virtual void updateAnno(const Annotation &anno) = 0;

  [[nodiscard]] auto color() const {
    if (m_inHover) [[unlikely]] {
      return COLOR_HOVER;
    }
    return m_color;
  }
  void setColor(const QColor &color) { m_color = color; }

  // Pen for drawing the shape outline
  [[nodiscard]] auto pen() const {
    QPen pen(color());
    pen.setWidth(PEN_WIDTH);
    return pen;
  }

  [[nodiscard]] auto text() const { return m_textGraphic->text(); }
  void setText(const QString &text) { m_textGraphic->setText(text); }

  [[nodiscard]] auto textItem() { return m_textGraphic; }

  void updateScaleFactor(double scaleFactor);

protected:
  void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
  void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

private:
  QGraphicsSimpleTextItem *m_textGraphic{}; // Child item
  QColor m_color;
  bool m_inHover{false};
};

} // namespace annotation