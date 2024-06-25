#include "FrameController.hpp"
#include "AScanPlot.hpp"
#include "CoregDisplay.hpp"
#include "datetime.hpp"
#include "strConvUtils.hpp"
#include <QDebug>
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
#include <chrono>
#include <filesystem>
#include <memory>
#include <numeric>
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <string>
#include <uspam/json.hpp>

FrameController::FrameController(ReconParamsController *paramsController,
                                 DataProcWorker *worker,
                                 CoregDisplay *coregDisplay, QWidget *parent)
    : QWidget(parent), m_reconParams(paramsController), m_worker(worker),
      m_coregDisplay(coregDisplay), m_AScanPlot(new AScanPlot(this)),
      m_btnPlayPause(new QPushButton("Play", this)),
      m_actOpenFileSelectDialog(new QAction(QIcon{}, "Open binfile")) {

  // Actions
  connect(m_actOpenFileSelectDialog, &QAction::triggered, this,
          &FrameController::openFileSelectDialog);

  // Result ready
  connect(worker, &DataProcWorker::resultReady, this,
          [this](std::shared_ptr<BScanData<DataProcWorker::FloatType>> data) {
            m_data = std::move(data);

            plotCurrentBScan();
            plotCurrentAScan();
          });

  // UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  {
    auto *hlayout = new QHBoxLayout;
    vlayout->addLayout(hlayout);

    // Frame num label and spinbox
    {
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
    }

    // Slider to select frame num in the sequence
    {
      m_frameSlider = new QSlider(Qt::Horizontal);
      hlayout->addWidget(m_frameSlider);
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

    // Play/pause button
    {
      hlayout->addWidget(m_btnPlayPause);
      m_btnPlayPause->setDisabled(true);

      connect(m_btnPlayPause, &QPushButton::clicked, this,
              [&] { updatePlayingState(!m_isPlaying); });
    }
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

  // Signals emittied from CoregDisplay
  {
    connect(m_coregDisplay, &CoregDisplay::AScanSelected, [this](int idx) {
      // The index received here is in canvas pixmap coordinates (i.e. doesn't
      // account for flip and rotation offset)
      m_AScanPlotIdx_canvas = idx;

      plotCurrentAScan();
    });
  }

  vlayout->addWidget(m_AScanPlot);
}

void FrameController::openFileSelectDialog() {
  const QString filename = QFileDialog::getOpenFileName(
      this, tr("Open Bin File"), QString(), tr("Binfiles (*.bin)"));

  acceptNewBinfile(filename);
}

void FrameController::acceptNewBinfile(const QString &filename) {
  // Update GUI
  updatePlayingState(false);

  // Emit signal
  if (!filename.isEmpty()) {
    emit sigBinfileSelected(filename);
  }

  // Try to load the annotation file
  m_binPath = qString2Path(filename);
  m_annoPath = m_binPath.parent_path() /
               (m_binPath.stem().string() + "_annotations.json");

  // Load if exists
  if (fs::exists(m_annoPath)) {
    m_doc.readFromFile(m_annoPath);
    loadFrameAnnotationsFromDocToModel(frameNum());
  } else {
    m_doc.init();
    m_doc.writeToFile(m_annoPath);
  }
}

int FrameController::frameNum() const {
  const auto val = m_frameNumSpinBox->value();
  assert(val == m_frameSlider->value());
  return val;
}

void FrameController::setFrameNum(int frame) {
  const auto oldFrame = frameNum();
  // Save old frames's labels
  saveFrameAnnotationsFromModelToDoc(oldFrame);

  // Load labels for new frame
  loadFrameAnnotationsFromDocToModel(frame);

  // Save doc to file
  m_doc.writeToFile(m_annoPath);

  // Update GUI
  m_frameNumSpinBox->setValue(frame);
  m_frameSlider->setValue(frame);
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

void FrameController::saveFrameAnnotationsFromModelToDoc(int frame) {
  auto *model = m_coregDisplay->model();
  m_doc.setAnnotationForFrame(frame, model->annotations());
}

void FrameController::loadFrameAnnotationsFromDocToModel(int frame) {
  auto *model = m_coregDisplay->model();
  model->setAnnotations(m_doc.getAnnotationForFrame(frame));
}

void FrameController::AScanIdxUpdated(int idx) {
  m_AScanPlotIdx = idx;
  plotCurrentAScan();
}

void FrameController::plotCurrentAScan() {

  // Correct for flip and rotation in the selected AScan idx
  // and store result in m_AScanPlotIdx
  {
    auto idx = m_AScanPlotIdx_canvas;
    if (uspam::recon::ReconParams::flip(m_data->frameIdx)) {
      idx -= m_reconParams->params.PA.rotateOffset;

      constexpr int AScansPerBScan = 1000;
      if (idx < 0) {
        idx += AScansPerBScan;
      } else if (idx >= AScansPerBScan) {
        idx -= AScansPerBScan;
      }

      idx = AScansPerBScan - 1 - idx;
    }

    emit statusMessage(
        QString("Select AScan: %1. Flip: %2")
            .arg(idx)
            .arg(uspam::recon::ReconParams::flip(m_data->frameIdx)));

    m_AScanPlotIdx = idx;
  }

  // Display aline
  const auto &rf = m_data->rf;

  arma::vec x(rf.n_rows, arma::fill::none);
  std::iota(x.begin(), x.end(), 0);
  const auto y = arma::conv_to<arma::vec>::from(rf.col(m_AScanPlotIdx));
  m_AScanPlot->plot(x, y);
}

void FrameController::plotCurrentBScan() {
  // Display images
  m_coregDisplay->imshow(m_data->PAUSradial_img, m_data->USradial_img,
                         m_data->fct);
}