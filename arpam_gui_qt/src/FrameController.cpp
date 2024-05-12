#include "FrameController.hpp"

#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QToolTip>
#include <QVBoxLayout>
#include <cassert>

FrameController::FrameController(QWidget *parent) : QWidget(parent) {
  {
    auto *vlayout = new QVBoxLayout;
    this->setLayout(vlayout);
    {
      auto *hlayout = new QHBoxLayout;
      vlayout->addLayout(hlayout);

      auto *btnPickFile = new QPushButton("Load bin file");
      hlayout->addWidget(btnPickFile);
      connect(btnPickFile, &QPushButton::clicked, this,
              &FrameController::s_openBinFile);

      auto *btnPlay = new QPushButton("Play");
      hlayout->addWidget(btnPlay);
      connect(btnPlay, &QPushButton::clicked, this, [&]() { emit play(); });

      auto *btnStopProcEarly = new QPushButton("Pause");
      hlayout->addWidget(btnStopProcEarly);
      connect(btnStopProcEarly, &QPushButton::clicked, this,
              [&]() { emit pause(); });
    }

    {
      auto *hlayout = new QHBoxLayout;
      vlayout->addLayout(hlayout);

      auto *frameNumLabel = new QLabel;
      frameNumLabel->setText("Frame num:");
      hlayout->addWidget(frameNumLabel);

      frameNumSpinBox = new QSpinBox;
      hlayout->addWidget(frameNumSpinBox);
      frameNumSpinBox->setDisabled(true);

      // Slider to select scan in the sequence
      frameSlider = new QSlider(Qt::Horizontal);
      vlayout->addWidget(frameSlider);
      frameSlider->setDisabled(true);
      frameSlider->setTickPosition(QSlider::TickPosition::TicksBelow);

      connect(frameSlider, &QSlider::sliderPressed, this, [&] {
        emit pause();
        QToolTip::showText(QCursor::pos(),
                           QString("%1").arg(frameSlider->value()), nullptr);
      });
      connect(frameSlider, &QSlider::sliderMoved, this, [&] {
        QToolTip::showText(QCursor::pos(),
                           QString("%1").arg(frameSlider->value()), nullptr);
      });
      connect(frameSlider, &QSlider::sliderReleased, this,
              [&] { emit frameNumUpdated(frameSlider->value()); });
    }
  }
}

void FrameController::s_openBinFile() {
  QString filename = QFileDialog::getOpenFileName(
      this, tr("Open Bin File"), QString(), tr("Binfiles (*.bin)"));

  if (!filename.isEmpty()) {
    qInfo() << "Selected binfile" << filename;
    emit openBinFile(filename);
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
