#pragma once

#include <QWidget>
#include <qcustomplot.h>

/*
 * A custom, QCustomPlot with some QoL features
 */
class CustomPlot : public QCustomPlot {
  Q_OBJECT
public:
  static constexpr int TICK_LENGTH = 5;
  static constexpr int SUBTICK_LENGTH = 3;

  explicit CustomPlot(QWidget *parent = nullptr);

private:
};