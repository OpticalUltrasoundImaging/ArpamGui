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
#include <QStandardPaths>
#include <QString>
#include <QTextStream>
#include <QToolTip>
#include <QVBoxLayout>
#include <Qt>
#include <QtLogging>
#include <cassert>
#include <filesystem>
#include <fmt/format.h>
#include <memory>
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <string>
#include <uspam/json.hpp>
#include <uspam/timeit.hpp>

FrameController::FrameController(RFProducerFile *rfProducerFile,
                                 ReconWorker *reconWorker,
                                 ReconParamsController *paramsController,
                                 AScanPlot *ascanPlot,
                                 CoregDisplay *coregDisplay, QWidget *parent)
    : QWidget(parent),

      m_producerFile(rfProducerFile), m_reconWorker(reconWorker),

      m_reconParams(paramsController),

      m_coregDisplay(coregDisplay), m_AScanPlot(ascanPlot),

      m_btnPlayPause(new QPushButton("Play")),
      m_btnExportFrame(new QPushButton("Export frame")),
      m_btnExportAllFrames(new QPushButton("Export all frames")),

      m_menuFrame(new QMenu("Frames", this)),
      m_actOpenFileSelectDialog(new QAction("Open binfile")),
      m_actCloseBinfile(new QAction("Close binfile")),

      m_actPlayPause(new QAction("Play/Pause")),
      m_actNextFrame(new QAction("Next Frame")),
      m_actPrevFrame(new QAction("Prev Frame")),

      m_actExportFrame(new QAction("Export Frame")),
      m_actExportAllFrames(new QAction("Export All Frames")),

      m_actSaveRF(new QAction("Save RF")),
      m_actSaveRadialImages(new QAction("Save radial images")),
      m_actSaveRectImages(new QAction("Save rect images")),
      m_actSaveRoiOnly(new QAction("Only save frames with annotations")),

      m_exportDirDefault(qString2Path(
          QStandardPaths::writableLocation(QStandardPaths::DesktopLocation)))

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

      connect(m_frameSlider, &QSlider::sliderReleased, this, [&] {
        const auto idx = m_frameSlider->value();
        emit sigFrameNumUpdated(idx);
      });
    }

    {
      auto *layout = new QHBoxLayout;
      hlayout->addLayout(layout);

      // Play/pause action and button
      {
        m_actPlayPause->setCheckable(true);
        m_actPlayPause->setShortcut({Qt::Key_Space});
        connect(m_actPlayPause, &QAction::triggered,
                [this](bool checked) { updatePlayingState(!m_isPlaying); });
        m_menuFrame->addAction(m_actPlayPause);

        layout->addWidget(m_btnPlayPause);
        connect(m_btnPlayPause, &QPushButton::clicked, m_actPlayPause,
                &QAction::trigger);
      }

      // Frame navigation actions
      {
        m_actPrevFrame->setShortcut({Qt::Key_Comma});
        connect(m_actPrevFrame, &QAction::triggered, this,
                &FrameController::prevFrame);
        m_menuFrame->addAction(m_actPrevFrame);

        m_actNextFrame->setShortcut({Qt::Key_Period});
        connect(m_actNextFrame, &QAction::triggered, this,
                &FrameController::nextFrame);
        m_menuFrame->addAction(m_actNextFrame);
      }

      // Button and action to export current frame
      m_menuFrame->addSection("Export");
      {
        m_actExportFrame->setShortcut(Qt::CTRL | Qt::Key_E);
        connect(m_actExportFrame, &QAction::triggered,
                [this]() { exportCurrentFrame(m_exportDirDefault); });
        m_menuFrame->addAction(m_actExportFrame);

        m_btnExportFrame->setEnabled(false);
        layout->addWidget(m_btnExportFrame);
        connect(m_btnExportFrame, &QPushButton::clicked, m_actExportFrame,
                &QAction::trigger);
      }

      // Button and action to export all frames
      {
        m_actExportAllFrames->setShortcut(Qt::CTRL | Qt::SHIFT | Qt::Key_E);
        connect(m_actExportAllFrames, &QAction::triggered, this,
                &FrameController::handleExportAllFramesBtnClick);
        m_menuFrame->addAction(m_actExportAllFrames);

        m_btnExportAllFrames->setEnabled(false);
        layout->addWidget(m_btnExportAllFrames);
        connect(m_btnExportAllFrames, &QPushButton::clicked,
                m_actExportAllFrames, &QAction::trigger);
      }

      // Actions for ExportSetting
      m_menuFrame->addSection("Export setting");
      {
        const auto connectCheckableActionAndBool = [this](QAction *act,
                                                          bool &val) {
          act->setCheckable(true);
          act->setChecked(val);
          connect(act, &QAction::triggered,
                  [this, &val](const bool checked) { val = checked; });
          m_menuFrame->addAction(act);
        };

        connectCheckableActionAndBool(m_actSaveRF, m_exportSetting.saveRF);
        connectCheckableActionAndBool(m_actSaveRadialImages,
                                      m_exportSetting.saveRadialImages);
        connectCheckableActionAndBool(m_actSaveRectImages,
                                      m_exportSetting.saveRectImages);
        connectCheckableActionAndBool(m_actSaveRoiOnly,
                                      m_exportSetting.saveRoiOnly);
      }
    }

    // Before a binfile is loaded, disable frame control
    m_frameSlider->setDisabled(true);
    m_btnPlayPause->setDisabled(true);
    m_menuFrame->setDisabled(true);
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

              // Update sequence name and canvas dipslay
              const auto seqName = path.parent_path().stem() / path.stem();
              m_sequenceName = path2QString(seqName);
              m_coregDisplay->setSequenceName(m_sequenceName);

              //  Update export path for exportAllFrames
              m_exportDir = m_exportDirDefault / seqName;
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
    connect(m_reconWorker, &ReconWorker::imagesReady, this,
            &FrameController::receiveNewFrame);

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
  m_btnExportFrame->setEnabled(true);
  m_btnExportAllFrames->setEnabled(true);
  m_frameSlider->setEnabled(true);
  m_menuFrame->setEnabled(true);

  m_actCloseBinfile->setEnabled(true);
}

void FrameController::closeBinfile() {
  m_producerFile->stopProducing();
  m_producerFile->closeBinfile();

  m_binPath.clear();
  m_annoPath.clear();

  m_btnPlayPause->setEnabled(false);
  m_frameSlider->setEnabled(false);
  m_menuFrame->setEnabled(false);

  m_actCloseBinfile->setEnabled(false);
}

void FrameController::receiveNewFrame(
    std::shared_ptr<BScanData<ArpamFloat>> data) {

  uspam::TimeIt timeit;

  // Update AScan plot. AScan plot keeps a ref of data
  m_AScanPlot->setData(data);
  m_AScanPlot->plotCurrentAScan();

  // Move the shared_ptr to m_data to elide 1 ref inc/dec
  m_data = std::move(data);

  // Update BScan plot
  plotCurrentBScan();

  // Update frame index display
  const auto idx = m_data->frameIdx;
  setFrameNum(idx);
  m_coregDisplay->setIdx(idx);

  // Display metrics
  {
    auto msg = QString("Frame %1/%2: ").arg(idx).arg(m_producerFile->size());
    QTextStream stream(&msg);
    stream << m_data->metrics;
    emit message(msg);
  }
}

void FrameController::setFrameNum(int frame) {
  saveCurrAnnotationAndLoadNewFrame(frame);

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

    if (m_exportingAllFrames) {
      // Update exporting all state
      handleExportAllFramesBtnClick();
    }
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
  const auto *model = m_coregDisplay->model();
  if (model->dirty()) {
    const auto &annotations = m_coregDisplay->model()->annotations();
    m_doc.setAnnotationForFrame(frame, annotations);
    return true;
  }
  return false;
}

void FrameController::loadFrameAnnotationsFromDocToModel(int frame) {
  auto *model = m_coregDisplay->model();
  model->setAnnotations(m_doc.getAnnotationForFrame(frame));
}

void FrameController::saveCurrAnnotationAndLoadNewFrame(int newFrame) {
  // Save old frames's labels
  if (const auto oldFrame = frameNum();
      saveFrameAnnotationsFromModelToDoc(oldFrame)) {
    // If any annotations are present, save doc to file
    m_doc.writeToFile(m_annoPath);
  }

  // Load labels for new frame
  loadFrameAnnotationsFromDocToModel(newFrame);
}

void FrameController::plotCurrentBScan() {
  // Display images
  m_coregDisplay->imshow(m_data->PAUSradial_img, m_data->US.radial_img,
                         m_data->spacingRadialUS);
}

void FrameController::exportCurrentFrame(const fs::path &exportDir) {
  uspam::TimeIt<false> timeit;
  const auto session = m_binPath.parent_path().stem().string();
  const auto sequence = m_binPath.stem().string();
  const auto dirname =
      (session + "_" + sequence + "_" + std::to_string(m_data->frameIdx));

  const auto savedirpath = exportDir / dirname;

  m_data->exportToFile(savedirpath, m_coregDisplay->model()->annotations(),
                       m_exportSetting);

  auto elapsed = timeit.get_ms();

  emit message(QString("Exported frame %1. Took %2 ms")
                   .arg(m_data->frameIdx)
                   .arg(elapsed));
}

void FrameController::handleExportAllFramesBtnClick() {
  if (m_exportingAllFrames) {
    // Currently exporting. Abort and restore
    updatePlayingState(false);
    m_exportingAllFrames = false;

    m_reconWorker->stopExportingFrames();

    m_btnExportAllFrames->setText("Export All Frames");
    m_actExportAllFrames->setText("Export All Frames");

    m_btnPlayPause->setEnabled(true);
    m_btnExportFrame->setEnabled(true);
    m_actPlayPause->setEnabled(true);
    m_actPrevFrame->setEnabled(true);
    m_actNextFrame->setEnabled(true);
    m_actExportFrame->setEnabled(true);
  } else {
    m_btnPlayPause->setEnabled(false);
    m_btnExportFrame->setEnabled(false);
    m_actPlayPause->setEnabled(false);
    m_actPrevFrame->setEnabled(false);
    m_actNextFrame->setEnabled(false);
    m_actExportFrame->setEnabled(false);

    // Not currently exporting. Start
    m_reconWorker->shouldExportFrames(m_exportDir, &m_doc, m_exportSetting);
    m_exportingAllFrames = true;
    fs::create_directories(m_exportDir);

    updatePlayingState(true);

    m_btnExportAllFrames->setText("Abort Export All");
    m_actExportAllFrames->setText("Abort Export All");
  }
}
