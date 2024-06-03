#include "MainWindow.hpp"
#include "About.hpp"
#include "FrameController.hpp"
#include "ReconParamsController.hpp"
#include <QDockWidget>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QScrollArea>
#include <QSlider>
#include <QToolBar>
#include <QVBoxLayout>
#include <QtDebug>
#include <QtLogging>
#include <format>
#include <opencv2/opencv.hpp>
#include <qboxlayout.h>
#include <qtoolbar.h>
#include <uspam/defer.h>

namespace {
void setGlobalStyle(QLayout *layout) {
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);
}
} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), textEdit(new QPlainTextEdit(this)),
      canvasLeft(new Canvas(this)), canvasRight(new Canvas(this)),
      worker(new DataProcWorker) {

  // Enable QStatusBar at the bottom of the MainWindow
  statusBar();

  /**
   * Setup worker thread
   */
  {
    worker->moveToThread(&workerThread);

    connect(&workerThread, &QThread::finished, worker, &QObject::deleteLater);

    connect(worker, &DataProcWorker::resultReady, this,
            &MainWindow::handleNewImages);
    connect(worker, &DataProcWorker::error, this, &MainWindow::logError);

    workerThread.start();
  }

  /**
   * Setup GUI
   */
  // Config params and testing dock
  auto *dockLayout = new QHBoxLayout;
  {
    auto *dock = new QDockWidget("Config Widget", this);
    auto *dockWidget = new QWidget;
    dockWidget->setLayout(dockLayout);
    dock->setWidget(dockWidget);
    dock->setFeatures(dock->features() ^ (QDockWidget::DockWidgetClosable |
                                          QDockWidget::DockWidgetFloatable));
    this->addDockWidget(Qt::TopDockWidgetArea, dock);

    // Error box
    dockLayout->addWidget(textEdit);
    textEdit->setReadOnly(true);
    textEdit->setPlainText("Application started.\n");

    textEdit->appendPlainText(ARPAM_GUI_ABOUT()());
  }

  // Frame controller
  {
    auto *frameController = new FrameController;
    dockLayout->addWidget(frameController);
    connect(frameController, &FrameController::binfileSelected, worker,
            &DataProcWorker::setBinfile);
    connect(frameController, &FrameController::frameNumUpdated, worker,
            &DataProcWorker::playOne);
    connect(frameController, &FrameController::playClicked, worker,
            &DataProcWorker::play);
    connect(frameController, &FrameController::pauseClicked, this,
            [&]() { worker->pause(); });

    connect(worker, &DataProcWorker::maxFramesChanged, frameController,
            &FrameController::updateMaxFrameNum);
    connect(worker, &DataProcWorker::frameIdxChanged, frameController,
            &FrameController::updateFrameNum);
    connect(worker, &DataProcWorker::finishedOneFile, frameController,
            &FrameController::updatePlayingStatePause);
  }

  // Recon parameters controller
  {
    auto *reconParamsController = new ReconParamsController;
    dockLayout->addWidget(reconParamsController);

    connect(reconParamsController, &ReconParamsController::paramsUpdated,
            [this](uspam::recon::ReconParams2 params,
                   uspam::io::IOParams ioparams) {
              // Update params
              this->worker->updateParams(params, ioparams);

              // Only invoke "replayOne" if not currently worker is not playing
              if (this->worker->isReady() && !this->worker->isPlaying()) {
                QMetaObject::invokeMethod(worker, &DataProcWorker::replayOne);

                // Save params to file
                this->worker->saveParamsToFile();
              }
            });

    connect(reconParamsController, &ReconParamsController::error, this,
            &MainWindow::logError);
  }

  // Exit button
  {
    auto *layout = new QVBoxLayout;
    dockLayout->addLayout(layout);

    auto *closeBtn = new QPushButton("Close");
    layout->addWidget(closeBtn);
    connect(closeBtn, &QPushButton::clicked, this, &QMainWindow::close);
    closeBtn->setObjectName("closeButton");

    auto *toggleFullscreenBtn = new QPushButton("Toggle Fullscreen");
    layout->addWidget(toggleFullscreenBtn);
    connect(toggleFullscreenBtn, &QPushButton::clicked, this, [this] {
      if (this->isFullScreen()) {
        this->setWindowState(Qt::WindowMaximized);
      } else {
        this->setWindowState(Qt::WindowFullScreen);
      }
    });
    toggleFullscreenBtn->setObjectName("toggleFullscreenButton");
  }
  // End dock config

  // Central scroll area
  auto *centralLayout = new QVBoxLayout;
  {
    auto *centralWidget = new QScrollArea(this);
    centralWidget->setWidgetResizable(true);
    this->setCentralWidget(centralWidget);
    centralWidget->setLayout(centralLayout);
  }

  // auto *modeSwitchButton = new QPushButton("Switch Mode", this);
  // connect(modeSwitchButton, &QPushButton::clicked, this,
  //         &MainWindow::switchMode);
  // layout->addWidget(modeSwitchButton);

  // Add mode views
  // stackedWidget->addWidget(new RealTimeView());
  // stackedWidget->addWidget(new PostProcessingView());

  //   layout->addWidget(modeSwitchButton);

  //// Define Actions
  // Action to set the cursor mode to line measure
  actCursorUndo = new QAction(QIcon(), "Undo");
  // Action to set the cursor mode to line measure
  actCursorLine = new QAction(QIcon(), "Line");
  // Action to set the cursor mode to box zoom
  actCursorZoom = new QAction(QIcon(), "Zoom");

  connect(actCursorUndo, &QAction::triggered, [=] {
    canvasLeft->undo();
    canvasRight->undo();
  });

  actCursorLine->setCheckable(true);
  defer { actCursorLine->trigger(); };
  connect(actCursorLine, &QAction::triggered, [=] {
    canvasLeft->setCursorMode(Canvas::CursorMode::LineMeasure);
    canvasRight->setCursorMode(Canvas::CursorMode::LineMeasure);

    actCursorLine->setChecked(true);
    actCursorZoom->setChecked(false);
  });

  actCursorZoom->setCheckable(true);
  connect(actCursorZoom, &QAction::triggered, [=] {
    canvasLeft->setCursorMode(Canvas::CursorMode::BoxZoom);
    canvasRight->setCursorMode(Canvas::CursorMode::BoxZoom);

    actCursorLine->setChecked(false);
    actCursorZoom->setChecked(true);
  });

  {
    auto *vlayout = new QVBoxLayout;
    centralLayout->addLayout(vlayout);

    // Toolbar
    {
      auto *toolbar = new QToolBar("Cursor type");
      vlayout->addWidget(toolbar);

      toolbar->addAction(actCursorUndo);
      toolbar->addSeparator();
      toolbar->addAction(actCursorLine);
      toolbar->addAction(actCursorZoom);
    }

    // Image Canvas
    {
      auto *layout = new QHBoxLayout;
      vlayout->addLayout(layout);

      canvasLeft->setName("US");
      canvasRight->setName("PAUS");

      for (auto *const canvas : {canvasLeft, canvasRight}) {
        layout->addWidget(canvas);
        canvas->setStyleSheet("border: 1px solid black");
        canvas->setDisabled(true);

        connect(canvas, &Canvas::error, this, &MainWindow::logError);

        connect(canvas, &Canvas::mouseMoved, this,
                [&](QPoint pos, double depth_mm) {
                  statusBar()->showMessage(
                      QString("Pos: (%1, %2), depth: %3 mm")
                          .arg(pos.x())
                          .arg(pos.y())
                          .arg(depth_mm));
                });
      }
    }
  }

  // Set global style
  setGlobalStyle(dockLayout);
  setGlobalStyle(centralLayout);
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

void MainWindow::logError(const QString &message) {
  textEdit->appendPlainText(message);
}

void MainWindow::switchMode() {
  //   int currentIndex = stackedWidget->currentIndex();
  //   stackedWidget->setCurrentIndex(1 - currentIndex); // Toggle between 0 and
  //   1
}

void MainWindow::handleNewImages(QImage img1, QImage img2, double pix2m) {
  canvasLeft->imshow(QPixmap::fromImage(std::move(img1)), pix2m);
  canvasRight->imshow(QPixmap::fromImage(std::move(img2)), pix2m);
  canvasLeft->setEnabled(true);
  canvasRight->setEnabled(true);
}