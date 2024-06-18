#pragma once

#include <QLabel>
#include <QSize>
#include <QWidget>

class CanvasOverlay : public QWidget {
public:
  explicit CanvasOverlay(QWidget *parent = nullptr);

  [[nodiscard]] auto sequence() const { return m_sequence->text(); }
  void setSequence(const QString &sequence) { m_sequence->setText(sequence); }

  [[nodiscard]] auto modality() const { return m_modality->text(); }
  void setModality(const QString &modality) { m_modality->setText(modality); }

  void setMaxIdx(int maxIdx);

  void setIdx(int idx);

  void setSize(const QSize &size);

  void setZoom(double zoom);

private:
  // Top left
  QLabel *m_sequence;

  // Bottom left
  QLabel *m_modality;
  QLabel *m_idxLabel;
  int m_maxIdx{};
  QLabel *m_size{};

  // Top right

  // Bottom right
  QLabel *m_zoom{};
};