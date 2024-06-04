#include "FrameController.hpp"

#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QSlider>
#include <QSpinBox>
#include <QToolTip>
#include <QVBoxLayout>
#include <cassert>
#include <qmessagebox.h>

FrameController::FrameController(QWidget *parent)
    : QGroupBox("Frame controller", parent),
      m_btnPlayPause(new QPushButton("Play", this)) {
  {
    auto *vlayout = new QVBoxLayout;
    this->setLayout(vlayout);
    {
      auto *hlayout = new QHBoxLayout;
      vlayout->addLayout(hlayout);

      auto *btnPickFile = new QPushButton("Load bin file");
      hlayout->addWidget(btnPickFile);
      connect(btnPickFile, &QPushButton::clicked, this, [&] {
        const QString filename = QFileDialog::getOpenFileName(
            this, tr("Open Bin File"), QString(), tr("Binfiles (*.bin)"));

        acceptNewBinfile(filename);
      });

      hlayout->addWidget(m_btnPlayPause);
      m_btnPlayPause->setEnabled(false);

      connect(m_btnPlayPause, &QPushButton::clicked, this,
              [&] { updatePlayingState(!m_isPlaying); });
    }

    {
      auto *hlayout = new QHBoxLayout;
      vlayout->addLayout(hlayout);

      auto *frameNumLabel = new QLabel;
      frameNumLabel->setText("Frame:");
      hlayout->addWidget(frameNumLabel);

      // SpinBox to display frame num
      m_frameNumSpinBox = new QSpinBox;
      hlayout->addWidget(m_frameNumSpinBox);
      m_frameNumSpinBox->setDisabled(true);
      connect(m_frameNumSpinBox, &QSpinBox::editingFinished, this, [&] {
        auto val = m_frameNumSpinBox->value();
        emit sigFrameNumUpdated(val);
        updatePlayingState(false);
      });

      // Slider to select frame num in the sequence
      m_frameSlider = new QSlider(Qt::Horizontal);
      vlayout->addWidget(m_frameSlider);
      m_frameSlider->setDisabled(true);
      m_frameSlider->setTickPosition(QSlider::TickPosition::TicksBelow);

      connect(m_frameSlider, &QSlider::sliderPressed, this, [&] {
        updatePlayingState(false);
        QToolTip::showText(QCursor::pos(),
                           QString::number(m_frameSlider->value()));
      });

      connect(m_frameSlider, &QSlider::sliderMoved, this, [&] {
        const auto val = m_frameSlider->value();
        QToolTip::showText(QCursor::pos(), QString::number(val));
        m_frameNumSpinBox->setValue(val);
      });

      connect(m_frameSlider, &QSlider::sliderReleased, this,
              [&] { emit sigFrameNumUpdated(m_frameSlider->value()); });
    }
  }
}

void FrameController::acceptNewBinfile(const QString &filename) {
  updatePlayingState(false);

  if (!filename.isEmpty()) {
    qInfo() << "Selected binfile" << filename;
    emit sigBinfileSelected(filename);

    m_btnPlayPause->setEnabled(true);
    updatePlayingState(true);
  }
}
void FrameController::updateFrameNum(int frameNum) {
  m_frameNumSpinBox->setValue(frameNum);
  m_frameSlider->setValue(frameNum);
}

void FrameController::updateMaxFrameNum(int maxFrameNum) {
  assert(maxFrameNum > 0);
  m_frameSlider->setMinimum(0);
  m_frameSlider->setMaximum(maxFrameNum - 1);
  m_frameNumSpinBox->setMinimum(0);
  m_frameNumSpinBox->setMaximum(maxFrameNum - 1);

  m_frameNumSpinBox->setEnabled(true);
  m_frameSlider->setEnabled(true);
}

void FrameController::updatePlayingState(bool playing) {
  m_isPlaying = playing;
  if (playing) {
    // Now playing
    m_btnPlayPause->setText("Pause");
    emit sigPlay();
  } else {
    // Now pausing
    m_btnPlayPause->setText("Play");
    emit sigPause();
  }
}