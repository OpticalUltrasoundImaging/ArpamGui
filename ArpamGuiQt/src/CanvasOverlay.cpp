#include "CanvasOverlay.hpp"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <Qt>

// Top left: sequence
// Top right: nothing
// Bottom right: zoom
// Bottom left: modality, idxLabel, size
CanvasOverlay::CanvasOverlay(QWidget *parent)
    : QWidget(parent), m_sequence(new QLabel), m_modality(new QLabel),
      m_idxLabel(new QLabel), m_size(new QLabel), m_zoom(new QLabel)

{

  setAttribute(Qt::WA_TransparentForMouseEvents);
  setAttribute(Qt::WA_NoSystemBackground, true);
  setAttribute(Qt::WA_TranslucentBackground, true);

  setWindowFlags(Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint |
                 Qt::WindowTransparentForInput);

  setStyleSheet("QLabel { color: white; border: none }");

  auto *layout = new QHBoxLayout;
  this->setLayout(layout);
  auto *left = new QVBoxLayout;
  auto *right = new QVBoxLayout;
  layout->addLayout(left);
  layout->addLayout(right);

  // Top left
  {
    auto *topLeft = new QVBoxLayout;
    topLeft->setAlignment(Qt::AlignTop);
    left->addLayout(topLeft);

    topLeft->addWidget(m_sequence);
  }

  {
    // Bottom left
    auto *bottomLeft = new QVBoxLayout;
    bottomLeft->setAlignment(Qt::AlignBottom);
    left->addLayout(bottomLeft);

    bottomLeft->addWidget(m_modality);
    bottomLeft->addWidget(m_idxLabel);
    bottomLeft->addWidget(m_size);
  }

  {
    // Top right
    auto *topRight = new QVBoxLayout;
    topRight->setAlignment(Qt::AlignTop | Qt::AlignRight);
    right->addLayout(topRight);
  }

  {
    // Bottom right
    auto *bottomRight = new QVBoxLayout;
    bottomRight->setAlignment(Qt::AlignBottom | Qt::AlignRight);
    right->addLayout(bottomRight);

    const std::array labels = {m_zoom};

    for (const auto &w : labels) {
      w->setAlignment(Qt::AlignRight);
      bottomRight->addWidget(w);
    }
  }
};

void CanvasOverlay::setMaxIdx(int maxIdx) { m_maxIdx = maxIdx; }

void CanvasOverlay::setIdx(int idx) {
  m_idxLabel->setText(QString("Img: %1/%2").arg(idx).arg(m_maxIdx));
}

void CanvasOverlay::setSize(const QSize &size) {
  m_size->setText(QString("%1 x %2").arg(size.width()).arg(size.height()));
}

void CanvasOverlay::setZoom(double zoom) {
  constexpr int fct2pct = 100;
  const int zoomPct = static_cast<int>(fct2pct * zoom);
  m_zoom->setText(QString("Zoom: %1%").arg(zoomPct));
}
