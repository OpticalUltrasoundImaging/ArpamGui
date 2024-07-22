#include "MainWindow.hpp"
#include "About.hpp"
#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
#include "FrameController.hpp"
#include "ReconParamsController.hpp"
#include <QAction>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QIcon>
#include <QKeySequence>
#include <QLabel>
#include <QMessageBox>
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

      worker(new DataProcWorker),

      textEdit(new QPlainTextEdit(this)),
      reconParamsController(new ReconParamsController),
      m_AScanPlot(new AScanPlot(reconParamsController)),
      m_coregDisplay(new CoregDisplay),
      m_frameController(new FrameController(reconParamsController, worker,
                                            m_AScanPlot, m_coregDisplay))

{
  menuBar()->addMenu(m_frameController->frameMenu());

  // Enable QStatusBar at the bottom of the MainWindow
  statusBar();

  // Enable drop (bin files)
  setAcceptDrops(true);

  /**
   * Setup worker thread
   */
  {
    worker->moveToThread(&workerThread);

    connect(&workerThread, &QThread::finished, worker, &QObject::deleteLater);

    connect(worker, &DataProcWorker::error, this, &MainWindow::logError);

    // Note: worker's resultReady signal is connected inside FrameController

    // Start the worker thread event loop
    workerThread.start();
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
    resizeDocks({dockFrameController},
                {dockFrameController->sizeHint().height()},
                Qt::Orientation::Vertical);

    // dockLayout->addWidget(m_frameController);
    dockFrameController->setWidget(m_frameController);
    m_fileMenu->addAction(m_frameController->get_actOpenFileSelectDialog());
    m_viewMenu->addAction(dockFrameController->toggleViewAction());

    connect(m_frameController, &FrameController::message, this,
            &MainWindow::logError);

    connect(m_frameController, &FrameController::statusMessage, statusBar(),
            &QStatusBar::showMessage);
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
              [this](uspam::recon::ReconParams2 params,
                     uspam::io::IOParams ioparams) {
                // Update params
                this->worker->updateParams(std::move(params), ioparams);

                // Only invoke "replayOne" if not currently worker is not
                // playing
                if (this->worker->isReady() && !this->worker->isPlaying()) {
                  QMetaObject::invokeMethod(worker, &DataProcWorker::replayOne);

                  // Save params to file
                  this->worker->saveParamsToFile();
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
    if (checked) {
      this->setWindowState(Qt::WindowFullScreen);
    } else {
      this->setWindowState(Qt::WindowMaximized);
    }
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
    m_frameController->acceptNewBinfile(filepath);

    event->acceptProposedAction();
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  // Stop the worker thread
  if (workerThread.isRunning()) {
    this->worker->pause();
    workerThread.quit();
    workerThread.wait();
  }
  event->accept();
}

void MainWindow::logError(QString message) {
  textEdit->appendPlainText(message);
}
