#include "MainWindow.hpp"
#include "ReconParamsController.hpp"
#include <QDockWidget>
#include <QFileDialog>
#include <QScrollArea>
#include <QSlider>
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

    connect(this, &MainWindow::setProcWorkerBinfile, worker,
            &DataProcWorker::setBinfile);

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

  // Post processing controller
  {
    auto *btnPickFile = new QPushButton("Load bin file");
    connect(btnPickFile, &QPushButton::clicked, this, &MainWindow::openBinFile);
    dockLayout->addWidget(btnPickFile);

    auto *btnStopProcEarly = new QPushButton("Stop");

    connect(btnStopProcEarly, &QPushButton::clicked, this,
            &MainWindow::abortCurrentWorkInThread);
    dockLayout->addWidget(btnStopProcEarly);

    // Controls to update the recon parameters

    // Slider to select scan in the sequence
    auto *scanSlider = new QSlider(Qt::Horizontal);
    dockLayout->addWidget(scanSlider);
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

    // auto pixmap = QPixmap(":/resources/images/radial_380.png");
    // canvasLeft->imshow(pixmap);
  }

  // Set global style
  setGlobalStyle(dockLayout);
  setGlobalStyle(centralLayout);
}

void MainWindow::closeEvent(QCloseEvent *event) {
  // Stop the worker thread
  if (workerThread.isRunning()) {
    abortCurrentWorkInThread();
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

void MainWindow::openBinFile() {

  QString filename = QFileDialog::getOpenFileName(
      this, tr("Open Bin File"), QString(), tr("Binfiles (*.bin)"));

  if (!filename.isEmpty()) {
    qInfo() << "Selected binfile" << filename;
    emit setProcWorkerBinfile(filename);
  }
}

void MainWindow::abortCurrentWorkInThread() {
  this->worker->abortCurrentWork();
}

void MainWindow::handleNewImages(QImage img1, QImage img2) {
  canvasLeft->imshow(img1);
  canvasRight->imshow(img2);
}