#include "MainWindow.hpp"
#include "About.hpp"
#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
#include "FrameController.hpp"
#include "ReconParamsController.hpp"
#include "strConvUtils.hpp"
#include <QAction>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QMimeData>
#include <QScrollArea>
#include <QSlider>
#include <QTabWidget>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWidget>
#include <QtDebug>
#include <QtLogging>
#include <filesystem>
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

      textEdit(new QPlainTextEdit(this)), m_coregDisplay(new CoregDisplay),
      m_frameController(new FrameController(worker, m_coregDisplay))

{
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

    connect(worker, &DataProcWorker::resultReady, m_coregDisplay,
            &CoregDisplay::imshow);

    connect(worker, &DataProcWorker::error, this, &MainWindow::logError);

    // Start the worker thread event loop
    workerThread.start();
  }

  /**
   * Setup GUI
   */

  // Log dock widget
  {
    auto *dock = new QDockWidget("Log", this);
    // dock->setFeatures(dock->features() ^ (QDockWidget::DockWidgetClosable));
    this->addDockWidget(Qt::TopDockWidgetArea, dock);
    m_viewMenu->addAction(dock->toggleViewAction());

    // Error box
    // dockLayout->addWidget(textEdit);
    dock->setWidget(textEdit);
    textEdit->setReadOnly(true);
    textEdit->setPlainText("Application started.\n");
    textEdit->appendPlainText(ARPAM_GUI_ABOUT()());
  }

  // Frame controller dock widget
  {
    auto *dock = new QDockWidget("Frame Controller", this);
    // dock->setFeatures(dock->features() ^ (QDockWidget::DockWidgetClosable));
    this->addDockWidget(Qt::TopDockWidgetArea, dock);
    resizeDocks({dock}, {dock->sizeHint().height()}, Qt::Orientation::Vertical);

    // dockLayout->addWidget(m_frameController);
    dock->setWidget(m_frameController);
    m_fileMenu->addAction(m_frameController->get_actOpenFileSelectDialog());
    m_viewMenu->addAction(dock->toggleViewAction());
  }

  // Tabify ReconParamsController dock and Annotations dock on the left
  {
    setTabPosition(Qt::LeftDockWidgetArea, QTabWidget::North);

    // Recon parameters controller
    QDockWidget *reconParamsDock{};
    {
      auto *dock = new QDockWidget("Recon Parameters", this);
      reconParamsDock = dock;
      // dock->setFeatures(dock->features() ^
      // (QDockWidget::DockWidgetClosable));
      this->addDockWidget(Qt::LeftDockWidgetArea, dock);
      m_viewMenu->addAction(dock->toggleViewAction());
      resizeDocks({dock}, {dock->sizeHint().width()},
                  Qt::Orientation::Horizontal);

      auto *reconParamsController = new ReconParamsController;
      // dockLayout->addWidget(reconParamsController);
      dock->setWidget(reconParamsController);

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
      auto *dock = new QDockWidget("Annotations", this);
      this->addDockWidget(Qt::LeftDockWidgetArea, dock);
      m_viewMenu->addAction(dock->toggleViewAction());

      // Tabify annotation view and ReconParamsController
      this->tabifyDockWidget(reconParamsDock, dock);

      dock->setWidget(m_coregDisplay->annotationView());
    }

    // By default the last added tabified widget (Annotations) is activated.
    // Manually activate reconParamsDock
    reconParamsDock->raise();
  }

  // Exit button
  {
    auto *dock = new QDockWidget("Exit", this);
    // dock->setFeatures(dock->features() ^ (QDockWidget::DockWidgetClosable));
    this->addDockWidget(Qt::TopDockWidgetArea, dock);
    m_viewMenu->addAction(dock->toggleViewAction());

    auto *w = new QWidget;
    auto *layout = new QVBoxLayout;
    // dockLayout->addLayout(layout);
    w->setLayout(layout);
    dock->setWidget(w);

    auto *closeBtn = new QPushButton("Close");
    layout->addWidget(closeBtn);
    connect(closeBtn, &QPushButton::clicked, this, &QMainWindow::close);
    closeBtn->setObjectName("closeButton");

    auto *toggleFullscreenBtn = new QPushButton("Toggle Fullscreen");
    layout->addWidget(toggleFullscreenBtn);
    connect(toggleFullscreenBtn, &QPushButton::clicked, this,
            &MainWindow::toggleFullScreen);
    toggleFullscreenBtn->setObjectName("toggleFullscreenButton");
  }
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

  // auto *modeSwitchButton = new QPushButton("Switch Mode", this);
  // connect(modeSwitchButton, &QPushButton::clicked, this,
  //         &MainWindow::switchMode);
  // layout->addWidget(modeSwitchButton);

  // Add mode views
  // stackedWidget->addWidget(new RealTimeView());
  // stackedWidget->addWidget(new PostProcessingView());

  //   layout->addWidget(modeSwitchButton);

  // Set global style
  setGlobalStyle(m_coregDisplay->layout());
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

void MainWindow::keyPressEvent(QKeyEvent *event) {
  const auto KeyNextFrame = Qt::Key_Period;
  const auto KeyPrevFrame = Qt::Key_Comma;
  const auto KeyPlayPause = Qt::Key_Space;

  if (event->modifiers().testFlag(Qt::ControlModifier)) {
    // Control modifier
    switch (event->key()) {
    case Qt::Key_F:
      toggleFullScreen();
      break;
    default:
      QMainWindow::keyPressEvent(event);
    }

  } else {
    // No modifier
    switch (event->key()) {

    case KeyNextFrame:
      m_frameController->nextFrame();
      break;
    case KeyPrevFrame:
      m_frameController->prevFrame();
      break;
    case KeyPlayPause:
      m_frameController->togglePlayPause();
      break;

      /* Hotkeys to trigger cursor modes */
    case Qt::Key_D:
      m_coregDisplay->actionDefault()->trigger();
      break;
    case Qt::Key_P:
      m_coregDisplay->actionPan()->trigger();
      break;
    case Qt::Key_L:
      m_coregDisplay->actionLine()->trigger();
      break;
    case Qt::Key_R:
      m_coregDisplay->actionLabelRect()->trigger();
      break;
    case Qt::Key_F:
      m_coregDisplay->actionLabelFan()->trigger();
      break;

    default:
      QMainWindow::keyPressEvent(event);
    }
  }
}

void MainWindow::contextMenuEvent(QContextMenuEvent *event) {
  // TODO
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

void MainWindow::toggleFullScreen() {
  if (this->isFullScreen()) {
    this->setWindowState(Qt::WindowMaximized);
  } else {
    this->setWindowState(Qt::WindowFullScreen);
  }
}