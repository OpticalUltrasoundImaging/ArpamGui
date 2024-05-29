#pragma once

#include <QLine>
#include <QPainter>
#include <vector>

struct ImshowCanvasTicks {
  static constexpr int margin = 5; // [px]

  template <bool isVertical> struct Ticks {
    static constexpr int minorTickSize = 10; // [px]
    static constexpr int majorTickSize = 15; // [px]

    // Display major tick every n ticks
    static constexpr int majorTickInterval = 10;

    std::vector<int> minorTicks; // position in m_pixmapScaled
    std::vector<QLine> minorTickLines;

    std::vector<int> majorTicks;
    std::vector<QLine> majorTickLines;

    std::vector<int> majorTickLabels;

    void clear() {
      minorTicks.clear();
      minorTickLines.clear();

      majorTicks.clear();
      majorTickLines.clear();
      majorTickLabels.clear();
    }

    void update(int size, double pix2mm) {
      // Draw divisions in increments of 1 mm
      const double divisionSize = 1 / pix2mm;

      const int numDivisions = static_cast<int>(size / (2 * divisionSize));
      // numDivisions = numDivisions - (numDivisions % displayInterval) + 1;

      for (int i = -numDivisions; i < numDivisions; ++i) {
        const auto pos = size / 2 + static_cast<int>(i * divisionSize);
        if (i % majorTickInterval == 0) {
          majorTicks.push_back(pos);
          majorTickLabels.push_back(abs(i));

          if constexpr (isVertical) {
            // Vertical scale bar
            majorTickLines.emplace_back(0, pos, minorTickSize, pos);
          } else {
            // Horizontal scale bar
            majorTickLines.emplace_back(pos, 0, pos, minorTickSize);
          }

        } else {
          minorTicks.push_back(pos);

          if constexpr (isVertical) {
            // Vertical scale bar
            minorTickLines.emplace_back(0, pos, minorTickSize, pos);
          } else {
            // Horizontal scale bar
            minorTickLines.emplace_back(pos, 0, pos, minorTickSize);
          }
        }
      }
    }
  };

  Ticks<true> vTicks;
  Ticks<false> hTicks;

  void clear() {
    hTicks.clear();
    vTicks.clear();
  }

  /**
   * @brief Update the state of the scalebar
   * @param pixmapSize size of the pixmap on which the scalebar will be drawn
   * @param pix2mm [mm] size of one pixel
   */
  void update(QSize pixmapSize, double pix2mm) {
    clear();
    hTicks.update(pixmapSize.width(), pix2mm);
    vTicks.update(pixmapSize.height(), pix2mm);
  }

  /**
   * @brief Draw the ticks onto a pixmap painter
   * @param painter
   */
  void draw(QPainter *painter) {
    // Draw minor ticks
    painter->setPen(Qt::gray);
    painter->drawLines(vTicks.minorTickLines.data(),
                       vTicks.minorTickLines.size());
    painter->drawLines(hTicks.minorTickLines.data(),
                       hTicks.minorTickLines.size());

    // Draw major ticks
    painter->setPen(Qt::white);
    painter->drawLines(vTicks.majorTickLines.data(),
                       vTicks.majorTickLines.size());
    painter->drawLines(hTicks.majorTickLines.data(),
                       hTicks.majorTickLines.size());

    for (int i = 0; i < vTicks.majorTicks.size(); ++i) {
      // Vertical scale bar (horizontal ticks)
      const auto pos = vTicks.majorTicks[i];
      painter->drawText(vTicks.majorTickSize + margin, pos + margin,
                        QString::number(vTicks.majorTickLabels[i]));
      // painter->drawText(pos - margin, majorTickSize + margin * 3,
    }
  }
};