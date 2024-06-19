#include "FrameController.hpp"
#include "CoregDisplay.hpp"
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QSlider>
#include <QSpinBox>
#include <QToolTip>
#include <QVBoxLayout>
#include <cassert>
#include <strConvUtils.hpp>

FrameController::FrameController(DataProcWorker *worker,
                                 CoregDisplay *coregDisplay, QWidget *parent)
    : QWidget(parent), m_worker(worker), m_coregDisplay(coregDisplay),
      m_btnPlayPause(new QPushButton("Play", this)),
      m_actOpenFileSelectDialog(new QAction(QIcon{}, "Open binfile")) {

  // Actions
  connect(m_actOpenFileSelectDialog, &QAction::triggered, this,
          &FrameController::openFileSelectDialog);

  // UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);
  {
    auto *hlayout = new QHBoxLayout;
    vlayout->addLayout(hlayout);

    auto *btnPickFile = new QPushButton("Load bin file");
    hlayout->addWidget(btnPickFile);
    connect(btnPickFile, &QPushButton::clicked, m_actOpenFileSelectDialog,
            &QAction::triggered);

    hlayout->addWidget(m_btnPlayPause);
    m_btnPlayPause->setDisabled(true);

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

  // Connections
  // Frame controller signals
  {

    // Action for when a new binfile is selected
    connect(this, &FrameController::sigBinfileSelected,
            [this](const QString &filepath) {
              const auto pathUtf8 = filepath.toUtf8();
              std::filesystem::path path(pathUtf8.constData());

              // Worker: load file
              QMetaObject::invokeMethod(m_worker, &DataProcWorker::setBinfile,
                                        path);

              // Update canvas dipslay
              m_coregDisplay->setSequenceName(path2QString(path.stem()));
            });

    // When frameController's changes it's frame number (through the drag bar
    // or play), tell the worker to process the right image
    connect(this, &FrameController::sigFrameNumUpdated, worker,
            &DataProcWorker::playOne);

    // Signal to start playing
    connect(this, &FrameController::sigPlay, this, [this] {
      // Invoke worker::play in the worker thread
      QMetaObject::invokeMethod(m_worker, &DataProcWorker::play);
      m_coregDisplay->resetZoomOnNextImshow();
    });

    // Signal to pause playing
    connect(this, &FrameController::sigPause, this, [&] {
      // worker::pause is thread safe
      m_worker->pause();
    });

    connect(m_worker, &DataProcWorker::maxFramesChanged, this,
            [this](int maxIdx) {
              this->setMaxFrameNum(maxIdx);
              m_coregDisplay->setMaxIdx(maxIdx);
            });

    connect(m_worker, &DataProcWorker::frameIdxChanged, this, [this](int idx) {
      this->setFrameNum(idx);
      m_coregDisplay->setIdx(idx);
    });

    connect(worker, &DataProcWorker::finishedPlaying, this,
            [this] { this->updatePlayingState(false); });
  }
}

void FrameController::openFileSelectDialog() {
  const QString filename = QFileDialog::getOpenFileName(
      this, tr("Open Bin File"), QString(), tr("Binfiles (*.bin)"));

  acceptNewBinfile(filename);
}

void FrameController::acceptNewBinfile(const QString &filename) {
  updatePlayingState(false);

  if (!filename.isEmpty()) {
    emit sigBinfileSelected(filename);
  }
}

int FrameController::frameNum() const {
  const auto val = m_frameNumSpinBox->value();
  assert(val == m_frameSlider->value());
  return val;
}

void FrameController::setFrameNum(int frameNum) {
  m_frameNumSpinBox->setValue(frameNum);
  m_frameSlider->setValue(frameNum);
}

int FrameController::maxFrameNum() const {
  const auto val = m_frameNumSpinBox->maximum();
  assert(val == m_frameSlider->maximum());
  return val;
}

void FrameController::setMaxFrameNum(int maxFrameNum) {
  assert(maxFrameNum > 0);
  m_frameSlider->setMinimum(0);
  m_frameSlider->setMaximum(maxFrameNum - 1);
  m_frameNumSpinBox->setMinimum(0);
  m_frameNumSpinBox->setMaximum(maxFrameNum - 1);

  m_btnPlayPause->setEnabled(true);
  m_frameNumSpinBox->setEnabled(true);
  m_frameSlider->setEnabled(true);
}

void FrameController::updatePlayingState(bool playing) {
  if (m_isPlaying == playing) {
    return;
  }

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

void FrameController::togglePlayPause() { updatePlayingState(!m_isPlaying); }

void FrameController::nextFrame() {
  updatePlayingState(false);
  const auto idx = m_frameNumSpinBox->value();
  const auto maxIdx = m_frameNumSpinBox->maximum();
  if (idx < maxIdx) {
    setFrameNum(idx + 1);
    emit sigFrameNumUpdated(idx + 1);
  }
}

void FrameController::prevFrame() {
  updatePlayingState(false);
  const auto idx = m_frameNumSpinBox->value();
  if (idx > 0) {
    setFrameNum(idx - 1);
    emit sigFrameNumUpdated(idx - 1);
  }
}
