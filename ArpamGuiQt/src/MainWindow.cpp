#include "MainWindow.hpp"
#include "About.hpp"
#include "CanvasAnnotationModel.hpp"
#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
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
#include <qobjectdefs.h>
#include <qscrollarea.h>
#include <qtoolbar.h>
#include <qwidget.h>
#include <uspam/defer.h>
#include <utility>

namespace {
void setGlobalStyle(QLayout *layout) {
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);
}
} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), textEdit(new QPlainTextEdit(this)),
      m_coregDisplay(new CoregDisplay), worker(new DataProcWorker) {

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
  }

  {
    // Error box
    dockLayout->addWidget(textEdit);
    textEdit->setReadOnly(true);
    textEdit->setPlainText("Application started.\n");
    textEdit->appendPlainText(ARPAM_GUI_ABOUT()());
  }

  // Frame controller
  {
    m_frameController = new FrameController;
    dockLayout->addWidget(m_frameController);
    connect(m_frameController, &FrameController::sigBinfileSelected, worker,
            &DataProcWorker::setBinfile);
    connect(m_frameController, &FrameController::sigFrameNumUpdated, worker,
            &DataProcWorker::playOne);
    connect(m_frameController, &FrameController::sigPlay, [=] {
      QMetaObject::invokeMethod(worker, &DataProcWorker::play);
      m_coregDisplay->resetZoomOnNextImshow();
    });
    connect(m_frameController, &FrameController::sigPause, this,
            [&]() { worker->pause(); });

    connect(worker, &DataProcWorker::maxFramesChanged, m_frameController,
            &FrameController::updateMaxFrameNum);
    connect(worker, &DataProcWorker::frameIdxChanged, m_frameController,
            &FrameController::updateFrameNum);
    connect(worker, &DataProcWorker::finishedPlaying,
            [=] { m_frameController->updatePlayingState(false); });
  }

  // Recon parameters controller
  {
    auto *reconParamsController = new ReconParamsController;
    dockLayout->addWidget(reconParamsController);

    connect(reconParamsController, &ReconParamsController::paramsUpdated,
            [this](uspam::recon::ReconParams2 params,
                   uspam::io::IOParams ioparams) {
              // Update params
              this->worker->updateParams(std::move(params), ioparams);

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
  setGlobalStyle(dockLayout);
  setGlobalStyle(m_coregDisplay->layout());
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
  if (event->mimeData()->hasUrls()) {
    event->acceptProposedAction();
  }
}

void MainWindow::dropEvent(QDropEvent *event) {
  const auto *mimeData = event->mimeData();
  if (mimeData->hasUrls()) {
    const auto urls = mimeData->urls();

    if (urls.size() > 1) {
      QMessageBox mbox;
      mbox.setText(
          QString("Please drop 1 file at at a time. Received %1 files.")
              .arg(urls.size()));
      mbox.exec();
    } else {
      const auto filepath = urls[0].toLocalFile();
      m_frameController->acceptNewBinfile(filepath);
    }
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

void MainWindow::switchMode() {
  //   int currentIndex = stackedWidget->currentIndex();
  //   stackedWidget->setCurrentIndex(1 - currentIndex); // Toggle between 0 and
  //   1
}
