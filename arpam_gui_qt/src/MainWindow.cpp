#include "MainWindow.hpp"
#include "FrameController.hpp"
#include "ReconParamsController.hpp"
#include <QDockWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QSlider>
#include <QVBoxLayout>
#include <QtDebug>
#include <QtLogging>
#include <format>
#include <opencv2/opencv.hpp>
namespace {
void setGlobalStyle(QLayout *layout) {
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);
}
} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), textEdit(new QPlainTextEdit(this)),
      canvasLeft(new ImshowCanvas(this)), canvasRight(new ImshowCanvas(this)),
      worker(new DataProcWorker) {

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
    textEdit->setPlainText("Application started...\n");
  }

  // Frame controller
  {
    auto *frameController = new FrameController;
    dockLayout->addWidget(frameController);
    // TODO connect signals
    connect(frameController, &FrameController::openBinFile, worker,
            &DataProcWorker::setBinfile);
    connect(frameController, &FrameController::frameNumUpdated, worker,
            &DataProcWorker::playOne);
    connect(frameController, &FrameController::play, worker,
            &DataProcWorker::play);
    connect(frameController, &FrameController::pause, this,
            [&]() { worker->pause(); });

    connect(worker, &DataProcWorker::updateMaxFrames, frameController,
            &FrameController::updateMaxFrameNum);
    connect(worker, &DataProcWorker::updateFrameIdx, frameController,
            &FrameController::updateFrameNum);
  }

  // Recon parameters controller
  {
    auto *reconParamsController = new ReconParamsController;
    dockLayout->addWidget(reconParamsController);

    // connect(reconParamsController, &ReconParamsController::paramsUpdated,
    //         worker, &DataProcWorker::updateParams);

    connect(reconParamsController, &ReconParamsController::paramsUpdated,
            [this](uspam::recon::ReconParams2 params,
                   uspam::io::IOParams ioparams) {
              this->worker->updateParams(std::move(params),
                                         std::move(ioparams));
            });

    connect(reconParamsController, &ReconParamsController::error, this,
            &MainWindow::logError);
  }

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

  // Image Canvas
  {
    // centralLayout->addWidget(coregDisplay);
    auto *layout = new QHBoxLayout;
    centralLayout->addLayout(layout);

    {
      canvasLeft->setStyleSheet("border: 1px solid black");
      layout->addWidget(canvasLeft);
      auto *sliderLeft = new QSlider(Qt::Vertical);
      layout->addWidget(sliderLeft);
    }

    {
      canvasRight->setStyleSheet("border: 1px solid black");
      layout->addWidget(canvasRight);
      auto *scrollBarRight = new QSlider(Qt::Vertical);
      layout->addWidget(scrollBarRight);
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

void MainWindow::handleNewImages(QImage img1, QImage img2) {
  canvasLeft->imshow(img1);
  canvasRight->imshow(img2);
}