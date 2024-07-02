#pragma once

#include <qcustomplot.h>

class CustomPlot : public QCustomPlot {
  Q_OBJECT

public:
  explicit CustomPlot(QWidget *parent = nullptr);

protected:
  void leaveEvent(QEvent *event) override;

signals:
  void mouseLeft();
};