#include "FrameController.hpp"

#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QSpinBox>
#include <QToolTip>
#include <QVBoxLayout>
#include <cassert>

FrameController::FrameController(QWidget *parent)
    : QGroupBox("Frame controller", parent),
      btnPlay(new QPushButton("Play", this)),
      btnPause(new QPushButton("Pause", this)) {
  {
    auto *vlayout = new QVBoxLayout;
    this->setLayout(vlayout);
    {
      auto *hlayout = new QHBoxLayout;
      vlayout->addLayout(hlayout);

      auto *btnPickFile = new QPushButton("Load bin file");
      hlayout->addWidget(btnPickFile);
      connect(btnPickFile, &QPushButton::clicked, this, [&] {
        updatePlayingState(false);
        emit pauseClicked();

        QString filename = QFileDialog::getOpenFileName(
            this, tr("Open Bin File"), QString(), tr("Binfiles (*.bin)"));

        if (!filename.isEmpty()) {
          qInfo() << "Selected binfile" << filename;
          emit binfileSelected(filename);
          updatePlayingState(true);
        }
      });

      hlayout->addWidget(btnPlay);
      btnPlay->setEnabled(false);

      hlayout->addWidget(btnPause);
      btnPause->setEnabled(false);

      connect(btnPlay, &QPushButton::clicked, this, [&]() {
        updatePlayingState(true);
        emit playClicked();
      });

      connect(btnPause, &QPushButton::clicked, this, [&]() {
        updatePlayingState(false);
        emit pauseClicked();
      });
    }

    {
      auto *hlayout = new QHBoxLayout;
      vlayout->addLayout(hlayout);

      auto *frameNumLabel = new QLabel;
      frameNumLabel->setText("Frame num:");
      hlayout->addWidget(frameNumLabel);

      // SpinBox to display frame num
      frameNumSpinBox = new QSpinBox;
      hlayout->addWidget(frameNumSpinBox);
      frameNumSpinBox->setDisabled(true);
      // frameNumSpinBox->setReadOnly(true);
      connect(frameNumSpinBox, &QSpinBox::editingFinished, this, [&] {
        auto val = frameNumSpinBox->value();
        emit frameNumUpdated(val);
        updatePlayingState(false);
      });

      // Slider to select frame num in the sequence
      frameSlider = new QSlider(Qt::Horizontal);
      vlayout->addWidget(frameSlider);
      frameSlider->setDisabled(true);
      frameSlider->setTickPosition(QSlider::TickPosition::TicksBelow);

      connect(frameSlider, &QSlider::sliderPressed, this, [&] {
        emit pauseClicked();
        updatePlayingState(false);
        QToolTip::showText(QCursor::pos(),
                           QString("%1").arg(frameSlider->value()), nullptr);
      });
      connect(frameSlider, &QSlider::sliderMoved, this, [&] {
        const auto val = frameSlider->value();
        QToolTip::showText(QCursor::pos(), QString("%1").arg(val), nullptr);
        frameNumSpinBox->setValue(val);
      });
      connect(frameSlider, &QSlider::sliderReleased, this,
              [&] { emit frameNumUpdated(frameSlider->value()); });
    }
  }
}

void FrameController::updateFrameNum(int frameNum) {
  frameNumSpinBox->setValue(frameNum);
  frameSlider->setValue(frameNum);
}

void FrameController::updateMaxFrameNum(int maxFrameNum) {
  assert(maxFrameNum > 0);
  frameSlider->setMinimum(0);
  frameSlider->setMaximum(maxFrameNum - 1);
  frameNumSpinBox->setMinimum(0);
  frameNumSpinBox->setMaximum(maxFrameNum - 1);

  frameNumSpinBox->setEnabled(true);
  frameSlider->setEnabled(true);
}

void FrameController::updatePlayingState(bool playing) {
  if (playing) {
    btnPlay->setEnabled(false);
    btnPause->setEnabled(true);
  } else {
    btnPlay->setEnabled(true);
    btnPause->setEnabled(false);
  }
}
void FrameController::updatePlayingStatePlay() { updatePlayingState(true); }
void FrameController::updatePlayingStatePause() { updatePlayingState(false); }