#include "MainWindow.hpp"
#include "About.hpp"
#include "AcquisitionController.hpp"
#include "Common.hpp"
#include "CoregDisplay.hpp"
#include "FrameController.hpp"
#include "QMessageBox"
#include "RFProducerFile.hpp"
#include "ReconParamsController.hpp"
#include "ReconWorker.hpp"
#include "strConvUtils.hpp"
#include <QAction>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QIcon>
#include <QKeySequence>
#include <QLabel>
#include <QMimeData>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSlider>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QWidget>
#include <Qt>
#include <QtDebug>
#include <QtLogging>
#include <memory>
#include <opencv2/opencv.hpp>
#include <qnamespace.h>
#include <uspam/defer.h>
#include <utility>

namespace {
void setGlobalStyle(QLayout *layout) {
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);
}
} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),

      m_fileMenu(menuBar()->addMenu(tr("&File"))),
      m_viewMenu(menuBar()->addMenu(tr("&View"))),

      buffer(std::make_shared<RFBuffer<ArpamFloat>>()),

      textEdit(new QPlainTextEdit(this)),
      reconParamsController(new ReconParamsController),

      // Producers
      rfProducerFile(
          new RFProducerFile(buffer, reconParamsController->ioparams)),

      // Consumer
      reconWorker(new ReconWorker(buffer)),

      m_AScanPlot(new AScanPlot(reconParamsController)),
      m_coregDisplay(new CoregDisplay),
      m_frameController(new FrameController(rfProducerFile, reconWorker,
                                            reconParamsController, m_AScanPlot,
                                            m_coregDisplay))

{
  menuBar()->addMenu(m_frameController->frameMenu());

  // Enable QStatusBar at the bottom of the MainWindow
  statusBar();

  // Enable drop (bin files)
  setAcceptDrops(true);

  /*
  Setup RF producer thread
  */
  {
    rfProducerFile->moveToThread(&producerThreadFile);
    producerThreadFile.start();
  }

  /*
  Setup RF consumer thread
  */
  {
    reconWorker->moveToThread(&workerThread);
    workerThread.start();
    QMetaObject::invokeMethod(reconWorker, &ReconWorker::start);
  }

  /*
  View mode actions

  - Simple mode: hide parameter tuning docks
  - Expert mode: most dock panels visible
   */
  {
    actViewSimple = new QAction("Physician view", this);
    actViewSimple->setCheckable(true);
    actViewSimple->setShortcut(Qt::CTRL | Qt::SHIFT | Qt::Key_1);
    connect(actViewSimple, &QAction::toggled, this, [this](bool checked) {
      if (checked) {
        dockLog->hide();
        dockFrameController->show();
        dockFrameController->resize(
            QSize{dockFrameController->width(),
                  dockFrameController->minimumHeight()});

        dockReconParams->hide();
        dockAnnotations->hide();
        dockAScanPlot->hide();

        dockAcquisitionController->hide();

        actViewSimple->setChecked(true);
        actViewExpert->setChecked(false);

        m_coregDisplay->actionShowUSCanvas()->setChecked(true);
        m_coregDisplay->resetZoom();
      }
    });

    actViewExpert = new QAction("Engineer view", this);
    actViewExpert->setCheckable(true);
    actViewExpert->setShortcut(Qt::CTRL | Qt::SHIFT | Qt::Key_2);
    connect(actViewExpert, &QAction::toggled, this, [this](bool checked) {
      if (checked) {
        dockLog->show();
        dockFrameController->show();
        dockReconParams->show();
        dockAnnotations->show();
        dockAScanPlot->show();

        dockReconParams->raise();

#ifdef ARPAM_HAS_ALAZAR
        dockAcquisitionController->show();

#else
        dockAcquisitionController->hide();
#endif

        actViewSimple->setChecked(false);
        actViewExpert->setChecked(true);

        m_coregDisplay->actionShowUSCanvas()->setChecked(false);
      }
    });

    m_viewMenu->addAction(actViewSimple);
    m_viewMenu->addAction(actViewExpert);
    m_viewMenu->addSeparator();
  }

  /**
   * Setup GUI
   */

  m_viewMenu->addMenu(m_coregDisplay->viewMenu());
  m_viewMenu->addSeparator();
  m_viewMenu->addMenu(m_coregDisplay->cursorMenu());
  m_viewMenu->addSeparator();

  // Log dock widget
  {
    dockLog = new QDockWidget("Log", this);
    // dock->setFeatures(dock->features() ^ (QDockWidget::DockWidgetClosable));
    this->addDockWidget(Qt::TopDockWidgetArea, dockLog);
    m_viewMenu->addAction(dockLog->toggleViewAction());

    // Error box
    // dockLayout->addWidget(textEdit);
    dockLog->setWidget(textEdit);
    textEdit->setReadOnly(true);
    textEdit->appendPlainText(arpam_about::aboutString());
  }

  // Frame controller dock widget
  {
    dockFrameController = new QDockWidget("Frame Controller", this);
    // dock->setFeatures(dock->features() ^ (QDockWidget::DockWidgetClosable));
    this->addDockWidget(Qt::TopDockWidgetArea, dockFrameController);

    // dockLayout->addWidget(m_frameController);
    dockFrameController->setWidget(m_frameController);

    m_fileMenu->addAction(m_frameController->actOpenFileSelectDialog());
    m_fileMenu->addAction(m_frameController->actCloseBinfile());

    m_viewMenu->addAction(dockFrameController->toggleViewAction());

    connect(m_frameController, &FrameController::message, this,
            &MainWindow::logError);

    connect(m_frameController, &FrameController::statusMessage, statusBar(),
            &QStatusBar::showMessage);
  }

  // Acquisition controller dock
  {
    dockAcquisitionController = new QDockWidget("Acquisition Controller", this);
    this->addDockWidget(Qt::TopDockWidgetArea, dockAcquisitionController);

    m_viewMenu->addAction(dockAcquisitionController->toggleViewAction());
    auto *acquisitionController = new AcquisitionController(buffer);
    dockAcquisitionController->setWidget(acquisitionController);

#ifdef ARPAM_HAS_ALAZAR
    connect(&acquisitionController->controller,
            &AcquisitionControllerObj::maxIndexChanged, m_frameController,
            &FrameController::setMaxFrameNum);
    connect(&acquisitionController->controller,
            &AcquisitionControllerObj::maxIndexChanged, m_coregDisplay,
            &CoregDisplay::setMaxIdx);

    connect(&acquisitionController->controller,
            &AcquisitionControllerObj::acquisitionFinished, this,
            [this, acquisitionController] {
              // Log event
              const auto &path =
                  acquisitionController->controller.daq().binpath();
              if (!path.empty()) {
                const auto strpath = path2QString(path);
                const auto msg =
                    QString("Finished acquiring to %1").arg(strpath);
                qInfo() << msg;
                statusBar()->showMessage(msg);
                logError(msg);

                // Load binfile in frame controller
                m_frameController->acceptBinfile(strpath);
              }
            });

    connect(&acquisitionController->controller,
            &AcquisitionControllerObj::acquisitionStarted,
            [this, acquisitionController] {
              // Disable frame controller
              m_frameController->setEnabled(false);

              // Status message about save/display
              const auto &path =
                  acquisitionController->controller.daq().binpath();

              const auto msg =
                  path.empty() ? "Display only"
                               : QString("Acquiring to ") + path2QString(path);
              qInfo() << msg;
              statusBar()->showMessage(msg);
            });

    {
      auto *act = acquisitionController->actShowMotorTestPanel();
      m_viewMenu->addAction(act);
    }
#endif // ARPAM_HAS_ALAZAR
  }

  // Tabify ReconParamsController dock and Annotations dock on the left
  {
    setTabPosition(Qt::LeftDockWidgetArea, QTabWidget::North);

    // Recon parameters controller
    {
      dockReconParams = new QDockWidget("Recon Parameters", this);
      // dock->setFeatures(dock->features() ^
      // (QDockWidget::DockWidgetClosable));
      this->addDockWidget(Qt::LeftDockWidgetArea, dockReconParams);
      dockReconParams->toggleViewAction()->setShortcut(
          {Qt::CTRL | Qt::SHIFT | Qt::Key_P});
      m_viewMenu->addAction(dockReconParams->toggleViewAction());
      resizeDocks({dockReconParams}, {dockReconParams->sizeHint().width()},
                  Qt::Orientation::Horizontal);

      // Wrap reconParamsController in a ScrollArea since it may overflow
      auto *reconParamsScrollArea = new QScrollArea;
      reconParamsScrollArea->setWidgetResizable(true);
      reconParamsScrollArea->setHorizontalScrollBarPolicy(
          Qt::ScrollBarAlwaysOff);
      reconParamsScrollArea->setWidget(reconParamsController);
      constexpr auto LeftDockMinWidth = 250;
      reconParamsScrollArea->setMinimumWidth(LeftDockMinWidth);
      dockReconParams->setWidget(reconParamsScrollArea);

      connect(reconParamsController, &ReconParamsController::paramsUpdated,
              this,
              [this](uspam::recon::ReconParams2 params,
                     uspam::io::IOParams ioparams) {
                // Update params
                rfProducerFile->setIOParams(ioparams);
                reconWorker->reconstructor().setParams(params, ioparams);

                m_coregDisplay->setAlinesPerBscan(ioparams.alinesPerBscan);

                // Only invoke "replayOne" if not currently worker is not
                // playing
                if (rfProducerFile->ready() && !rfProducerFile->producing()) {
                  QMetaObject::invokeMethod(rfProducerFile,
                                            &RFProducerFile::reproduceOne);
                }
              });

      connect(reconParamsController, &ReconParamsController::error, this,
              &MainWindow::logError);
    }

    // Annotation view dock
    {
      dockAnnotations = new QDockWidget("Annotations", this);
      this->addDockWidget(Qt::LeftDockWidgetArea, dockAnnotations);
      dockAnnotations->toggleViewAction()->setShortcut(
          {Qt::CTRL | Qt::SHIFT | Qt::Key_L});
      m_viewMenu->addAction(dockAnnotations->toggleViewAction());

      // Tabify annotation view and ReconParamsController
      this->tabifyDockWidget(dockReconParams, dockAnnotations);

      dockAnnotations->setWidget(m_coregDisplay->annotationView());
    }

    // By default the last added tabified widget (Annotations) is activated.
    // Manually activate reconParamsDock
    dockReconParams->raise();
  }

  {
    dockAScanPlot = new QDockWidget("AScan Plot", this);
    this->addDockWidget(Qt::RightDockWidgetArea, dockAScanPlot);
    dockAScanPlot->toggleViewAction()->setShortcut(
        {Qt::CTRL | Qt::SHIFT | Qt::Key_A});
    m_viewMenu->addAction(dockAScanPlot->toggleViewAction());

    dockAScanPlot->setWidget(m_AScanPlot);
  }

  auto *fullscreenAction = new QAction("Full Screen");
  fullscreenAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_F));
  fullscreenAction->setCheckable(true);
  connect(fullscreenAction, &QAction::triggered, [this](bool checked) {
    this->setWindowState(checked ? Qt::WindowFullScreen : Qt::WindowMaximized);
  });
  m_viewMenu->addAction(fullscreenAction);

  // End dock config

  // Coreg display
  setCentralWidget(m_coregDisplay);

  connect(m_coregDisplay, &CoregDisplay::message, this, &MainWindow::logError);
  connect(m_coregDisplay, &CoregDisplay::mouseMoved, this,
          [&](QPoint pos, double depth_mm) {
            auto msg = QString("Pos: (%1, %2), depth: %3 mm")
                           .arg(pos.x())
                           .arg(pos.y())
                           .arg(depth_mm);
          });

  // Set global style
  setGlobalStyle(m_coregDisplay->layout());

  // About action
  const auto title = QString("About %1").arg(QApplication::applicationName());
  auto *actAbout = new QAction(title, this);
  actAbout->setMenuRole(QAction::AboutRole);
  statusBar()->showMessage(title);
  connect(actAbout, &QAction::triggered, this,
          [this] { arpam_about::showAboutDialog(this); });
  m_fileMenu->addAction(actAbout);

  actViewExpert->trigger();
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
  const auto *mimeData = event->mimeData();
  if (mimeData->hasUrls()) {
    const auto &urls = mimeData->urls();
    // Only allow dropping a single file
    if (urls.size() == 1) {
      const auto filepath = urls[0].toLocalFile();

      // Only allow a specific extension
      if (filepath.endsWith(".bin")) {
        event->acceptProposedAction();
      }
    }
  }
}

void MainWindow::dropEvent(QDropEvent *event) {
  const auto *mimeData = event->mimeData();
  if (mimeData->hasUrls()) {
    const auto &urls = mimeData->urls();
    const auto filepath = urls[0].toLocalFile();
    m_frameController->acceptBinfile(filepath);

    event->acceptProposedAction();
  }
}

void MainWindow::logError(QString message) {
  textEdit->appendPlainText(message);
}

void MainWindow::messageBox(const QString &title, const QString &message) {
  QMessageBox::information(this, title, message);
};

MainWindow::~MainWindow() {}

void MainWindow::closeEvent(QCloseEvent *event) {
  buffer->exit();

  if (workerThread.isRunning()) {
    // To signal recon worker to stop, call buffer->exit()
    workerThread.quit();
  }

  if (producerThreadFile.isRunning()) {
    // Signal producer objects to stop
    rfProducerFile->stopProducing();
    producerThreadFile.quit();
  }

  // Wait for threads to join
  producerThreadFile.wait();
  workerThread.wait();
}
