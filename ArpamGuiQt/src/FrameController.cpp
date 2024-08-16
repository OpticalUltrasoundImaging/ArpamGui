#include "FrameController.hpp"
#include "AScanPlot.hpp"
#include "Common.hpp"
#include "CoregDisplay.hpp"
#include "ReconParamsController.hpp"
#include "ReconWorker.hpp"
#include "strConvUtils.hpp"
#include <QDebug>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QMenu>
#include <QPlainTextEdit>
#include <QSlider>
#include <QSpinBox>
#include <QString>
#include <QTextStream>
#include <QToolTip>
#include <QVBoxLayout>
#include <Qt>
#include <QtLogging>
#include <cassert>
#include <filesystem>
#include <memory>
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <uspam/json.hpp>

FrameController::FrameController(

    RFProducerFile *rfProducerFile, ReconWorker *reconWorker,

    ReconParamsController *paramsController, AScanPlot *ascanPlot,
    CoregDisplay *coregDisplay, QWidget *parent)
    : QWidget(parent),

      m_producerFile(rfProducerFile), m_reconWorker(reconWorker),

      m_reconParams(paramsController),

      m_coregDisplay(coregDisplay), m_AScanPlot(ascanPlot),
      m_btnPlayPause(new QPushButton("Play", this)),

      m_menu(new QMenu("Frames", this)),
      m_actOpenFileSelectDialog(new QAction("Open binfile")),
      m_actCloseBinfile(new QAction("Close binfile")),

      m_actPlayPause(new QAction("Play/Pause")),
      m_actNextFrame(new QAction("Next Frame")),
      m_actPrevFrame(new QAction("Prev Frame"))

{

  // Actions
  m_actOpenFileSelectDialog->setShortcut({Qt::CTRL | Qt::Key_O});
  connect(m_actOpenFileSelectDialog, &QAction::triggered, this,
          &FrameController::openFileSelectDialog);

  m_actCloseBinfile->setEnabled(false);
  m_actCloseBinfile->setShortcut(Qt::CTRL | Qt::Key_W);
  connect(m_actCloseBinfile, &QAction::triggered, this,
          &FrameController::closeBinfile);

  // UI
  auto *vlayout = new QVBoxLayout;
  this->setLayout(vlayout);

  {
    auto *hlayout = new QHBoxLayout;
    vlayout->addLayout(hlayout);

    // Slider to select frame num in the sequence
    {
      m_frameSlider = new QSlider(Qt::Horizontal);
      hlayout->addWidget(m_frameSlider);
      m_frameSlider->setTickPosition(QSlider::TickPosition::TicksBelow);

      connect(m_frameSlider, &QSlider::sliderPressed, this, [&] {
        updatePlayingState(false);
        QToolTip::showText(QCursor::pos(),
                           QString::number(m_frameSlider->value()));
      });

      connect(m_frameSlider, &QSlider::sliderMoved, this, [&] {
        const auto val = m_frameSlider->value();
        QToolTip::showText(QCursor::pos(), QString::number(val));
      });

      connect(m_frameSlider, &QSlider::sliderReleased, this,
              [&] { emit sigFrameNumUpdated(m_frameSlider->value()); });
    }

    // Play/pause action and button
    {
      m_actPlayPause->setCheckable(true);
      m_actPlayPause->setShortcut({Qt::Key_Space});
      connect(m_actPlayPause, &QAction::triggered,
              [this](bool checked) { updatePlayingState(!m_isPlaying); });
      m_menu->addAction(m_actPlayPause);

      hlayout->addWidget(m_btnPlayPause);
      connect(m_btnPlayPause, &QPushButton::clicked, m_actPlayPause,
              &QAction::trigger);
    }

    // Frame navigation actions
    {
      m_actPrevFrame->setShortcut({Qt::Key_Comma});
      connect(m_actPrevFrame, &QAction::triggered, this,
              &FrameController::prevFrame);
      m_menu->addAction(m_actPrevFrame);

      m_actNextFrame->setShortcut({Qt::Key_Period});
      connect(m_actNextFrame, &QAction::triggered, this,
              &FrameController::nextFrame);
      m_menu->addAction(m_actNextFrame);
    }

    // Before a binfile is loaded, disable frame control
    m_frameSlider->setDisabled(true);
    m_btnPlayPause->setDisabled(true);
    m_menu->setDisabled(true);
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
              QMetaObject::invokeMethod(m_producerFile,
                                        &RFProducerFile::setBinfile, path);

              // Update canvas dipslay
              {
                const auto seq =
                    path2QString(path.parent_path().stem() / path.stem());
                m_coregDisplay->setSequenceName(seq);
              }
            });

    // When frameController's changes it's frame number (through the drag bar
    // or play), tell the worker to process the right image
    connect(this, &FrameController::sigFrameNumUpdated, rfProducerFile,
            &RFProducerFile::produceOne);

    connect(m_producerFile, &RFProducerFile::maxFramesChanged, this,
            &FrameController::setMaxFrameNum);

    connect(m_producerFile, &RFProducerFile::maxFramesChanged, m_coregDisplay,
            &CoregDisplay::setMaxIdx);

    // Result ready
    connect(
        m_reconWorker, &ReconWorker::imagesReady, this,
        [this](std::shared_ptr<BScanData<ArpamFloat>> data) {
          m_AScanPlot->setData(data);
          m_AScanPlot->plotCurrentAScan();

          m_data = std::move(data);
          plotCurrentBScan();

          const auto idx = m_data->frameIdx;
          this->setFrameNum(idx);
          m_coregDisplay->setIdx(idx);
          // qDebug() << "FrameController received idx =" << idx;

          // Display metrics
          {
            auto msg =
                QString("Frame %1/%2: ").arg(idx).arg(m_producerFile->size());
            QTextStream stream(&msg);
            stream << m_data->metrics;
            emit message(msg);
          }
        });

    connect(m_producerFile, &RFProducerFile::finishedProducing, this,
            [this] { this->updatePlayingState(false); });
  }

  // Signals emittied from CoregDisplay
  {
    connect(m_coregDisplay, &CoregDisplay::AScanSelected, m_AScanPlot,
            &AScanPlot::handleAScanSelected);
  }
}

void FrameController::openFileSelectDialog() {
  const QString filename = QFileDialog::getOpenFileName(
      this, tr("Open Bin File"), QString(), tr("Binfiles (*.bin)"));

  acceptBinfile(filename);
}

void FrameController::acceptBinfile(const QString &filename) {
  // Update GUI
  setEnabled(true);
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
  }

  m_btnPlayPause->setEnabled(true);
  m_frameSlider->setEnabled(true);
  m_menu->setEnabled(true);

  m_actCloseBinfile->setEnabled(true);
}

void FrameController::closeBinfile() {
  m_producerFile->stopProducing();
  m_producerFile->closeBinfile();

  m_binPath.clear();
  m_annoPath.clear();

  m_btnPlayPause->setEnabled(false);
  m_frameSlider->setEnabled(false);
  m_menu->setEnabled(false);

  m_actCloseBinfile->setEnabled(false);
}

int FrameController::frameNum() const {
  // const auto val = m_frameNumSpinBox->value();
  return m_frameSlider->value();
}

void FrameController::setFrameNum(int frame) {
  const auto oldFrame = frameNum();

  // Save old frames's labels
  if (saveFrameAnnotationsFromModelToDoc(oldFrame)) {
    // If any annotations are present, save doc to file
    m_doc.writeToFile(m_annoPath);
  }

  // Load labels for new frame
  loadFrameAnnotationsFromDocToModel(frame);

  // Update GUI
  m_frameSlider->setValue(frame);
}

int FrameController::maxFrameNum() const { return m_frameSlider->maximum(); }

void FrameController::setMaxFrameNum(int maxFrameNum) {
  assert(maxFrameNum > 0);
  m_frameSlider->setMinimum(0);
  m_frameSlider->setMaximum(maxFrameNum - 1);
}

void FrameController::updatePlayingState(bool playing) {
  if (m_isPlaying == playing) {
    return;
  }

  m_isPlaying = playing;
  if (playing) {
    // Now playing
    m_btnPlayPause->setText("Pause");

    // Invoke worker::play in the worker thread
    QMetaObject::invokeMethod(m_producerFile, &RFProducerFile::beginProducing);
    m_coregDisplay->resetZoomOnNextImshow();

  } else {
    // Now pausing
    m_btnPlayPause->setText("Play");

    m_producerFile->stopProducing();
  }
}

void FrameController::togglePlayPause() { updatePlayingState(!m_isPlaying); }

void FrameController::nextFrame() {
  updatePlayingState(false);
  const auto idx = m_frameSlider->value();
  const auto maxIdx = m_frameSlider->maximum();
  if (idx < maxIdx) {
    setFrameNum(idx + 1);
    emit sigFrameNumUpdated(idx + 1);
  }
}

void FrameController::prevFrame() {
  updatePlayingState(false);
  const auto idx = m_frameSlider->value();
  if (idx > 0) {
    setFrameNum(idx - 1);
    emit sigFrameNumUpdated(idx - 1);
  }
}

bool FrameController::saveFrameAnnotationsFromModelToDoc(int frame) {
  const auto &annotations = m_coregDisplay->model()->annotations();
  if (!annotations.isEmpty()) {
    m_doc.setAnnotationForFrame(frame, annotations);
    return true;
  }
  return false;
}

void FrameController::loadFrameAnnotationsFromDocToModel(int frame) {
  auto *model = m_coregDisplay->model();
  model->setAnnotations(m_doc.getAnnotationForFrame(frame));
}

void FrameController::plotCurrentBScan() {
  // Display images
  m_coregDisplay->imshow(m_data->PAUSradial_img, m_data->US.radial_img,
                         m_data->fct);
}
