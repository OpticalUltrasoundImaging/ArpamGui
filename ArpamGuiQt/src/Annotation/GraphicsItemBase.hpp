#pragma once

#include "Annotation/AnnotationModel.hpp"
#include "geometryUtils.hpp"
#include <QColor>
#include <QGraphicsItem>
#include <QPen>
#include <QPointF>
#include <Qt>
#include <utility>

namespace annotation {

class GraphicsItemBase : public QGraphicsItem {
public:
  inline const static QColor COLOR_HOVER = Qt::green;
  inline const static int PEN_WIDTH = 2;
  inline const static QColor COLOR_DEFAULT = Qt::white;

  GraphicsItemBase();
  explicit GraphicsItemBase(const Annotation &annotation,
                            QGraphicsItem *parent = nullptr);
  explicit GraphicsItemBase(const QColor &color,
                            QGraphicsItem *parent = nullptr);
  virtual ~GraphicsItemBase() override;

  virtual void updateAnno(const Annotation &anno) = 0;

  [[nodiscard]] auto getPen() const {
    if (m_inHover) [[unlikely]] {
      QPen pen(COLOR_HOVER);
      pen.setWidth(PEN_WIDTH);
      return pen;
    } else {
      QPen pen(m_color);
      pen.setWidth(PEN_WIDTH);
      return pen;
    }
  }

  [[nodiscard]] auto color() const { return m_color; }
  void setColor(const QColor &color) { m_color = color; }

  [[nodiscard]] auto name() const { return m_name; }
  void setName(QString name) { m_name = std::move(name); }

protected:
  void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
  void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

private:
  QColor m_color;
  QString m_name;
  bool m_inHover{false};
};

} // namespace annotation