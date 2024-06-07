#pragma once

#include "geometryUtils.hpp"
#include <QGraphicsItem>
#include <QGraphicsRectItem>
#include <QLineF>
#include <QPainter>
#include <QPen>
#include <QPointF>
#include <QRectF>
#include <QTransform>
#include <QVariant>
#include <array>
#include <vector>

class LineAnnotation : public QGraphicsItem {
public:
  LineAnnotation(const QPointF &start, const QPointF &end, const QColor &color,
                 QGraphicsItem *parent = nullptr)
      : QGraphicsItem(parent), startPoint(start), endPoint(end),
        lineColor(color) {
    setAcceptHoverEvents(true);
  }

  [[nodiscard]] QRectF boundingRect() const override {
    return QRectF(startPoint, endPoint).normalized();
  }

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override {
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setPen(QPen(lineColor));
    painter->drawLine(startPoint, endPoint);
  }

protected:
  void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override {
    Q_UNUSED(event)
    lineColor = Qt::red; // Change color on hover
    update();
  }

  void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override {
    Q_UNUSED(event)
    lineColor = Qt::blue; // Change back color when not hovering
    update();
  }

private:
  QPointF startPoint, endPoint;
  QColor lineColor;
};

class BoxAnnotation : public QGraphicsItem {
public:
  BoxAnnotation(const QRectF &rect, const QColor &color,
                QGraphicsItem *parent = nullptr)
      : QGraphicsItem(parent), boxRect(rect), boxColor(color) {
    setAcceptHoverEvents(true);
  }

  [[nodiscard]] QRectF boundingRect() const override { return boxRect; }

  void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
             QWidget *widget) override {
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setPen(QPen(boxColor));
    painter->drawRect(boxRect);
  }

protected:
  void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override {
    Q_UNUSED(event)
    boxColor = Qt::red; // Change color on hover
    update();
  }

  void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override {
    Q_UNUSED(event)
    boxColor = Qt::blue; // Change back color when not hovering
    update();
  }

private:
  QRectF boxRect;
  QColor boxColor;
};
