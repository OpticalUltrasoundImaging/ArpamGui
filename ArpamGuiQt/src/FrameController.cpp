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
      m_btnPlay(new QPushButton("Play", this)),
      m_btnPause(new QPushButton("Pause", this)) {
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

      hlayout->addWidget(m_btnPlay);
      m_btnPlay->setEnabled(false);

      hlayout->addWidget(m_btnPause);
      m_btnPause->setEnabled(false);

      connect(m_btnPlay, &QPushButton::clicked, this, [&] {
        updatePlayingState(true);
        emit playClicked();
      });

      connect(m_btnPause, &QPushButton::clicked, this, [&] {
        updatePlayingState(false);
        emit pauseClicked();
      });
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
        emit frameNumUpdated(val);
        updatePlayingState(false);
      });

      // Slider to select frame num in the sequence
      m_frameSlider = new QSlider(Qt::Horizontal);
      vlayout->addWidget(m_frameSlider);
      m_frameSlider->setDisabled(true);
      m_frameSlider->setTickPosition(QSlider::TickPosition::TicksBelow);

      connect(m_frameSlider, &QSlider::sliderPressed, this, [&] {
        emit pauseClicked();
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
              [&] { emit frameNumUpdated(m_frameSlider->value()); });
    }
  }
}

void FrameController::acceptNewBinfile(const QString &filename) {
  updatePlayingState(false);
  emit pauseClicked();

  if (!filename.isEmpty()) {
    qInfo() << "Selected binfile" << filename;
    emit binfileSelected(filename);
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
  if (playing) {
    m_btnPlay->setEnabled(false);
    m_btnPause->setEnabled(true);
  } else {
    m_btnPlay->setEnabled(true);
    m_btnPause->setEnabled(false);
  }
}
void FrameController::updatePlayingStatePlay() { updatePlayingState(true); }
void FrameController::updatePlayingStatePause() { updatePlayingState(false); }