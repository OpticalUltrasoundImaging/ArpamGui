#pragma once

#include <qcustomplot.h>

inline void setupAxis(QCPAxis *axis, const QString &label = {},
                      bool tickLabels = true, int tickInside = 0,
                      int tickOutside = 0, int subTickInside = 0,
                      int subTickOutside = 0) {
  axis->setLabel(label);
  axis->setTickLabels(tickLabels);
  axis->setTickLength(tickInside, tickOutside);
  axis->setSubTickLength(subTickInside, subTickOutside);
}