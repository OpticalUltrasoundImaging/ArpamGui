#include "MainWindow.hpp"
#include <QDockWidget>
#include <QFileDialog>
#include <QScrollArea>
#include <QtDebug>
#include <QtLogging>
#include <format>
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), textEdit(new QPlainTextEdit(this)),
      coregDisplay(new CoregDisplay(this)), worker(new DataProcWorker) {

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
  auto *dock_layout = new QHBoxLayout;
  {
    auto *dock = new QDockWidget("Config Widget", this);
    auto *dock_widget = new QWidget;
    dock_widget->setLayout(dock_layout);
    dock->setWidget(dock_widget);
    dock->setFeatures(dock->features() ^ (QDockWidget::DockWidgetClosable |
                                          QDockWidget::DockWidgetFloatable));
    this->addDockWidget(Qt::TopDockWidgetArea, dock);

    dock_layout->addWidget(new QLabel("This is the config dock."));

    // Error box
    dock_layout->addWidget(textEdit);
    textEdit->setReadOnly(true);
    textEdit->setPlainText("Application started...\n");
  }

  // Post processing controller
  {
    auto *btnPickFile = new QPushButton("Load bin file");
    connect(btnPickFile, &QPushButton::clicked, this, &MainWindow::openBinFile);
    dock_layout->addWidget(btnPickFile);

    auto *btnStopProcEarly = new QPushButton("Stop");

    connect(btnStopProcEarly, &QPushButton::clicked, this,
            &MainWindow::abortCurrentWorkInThread);
    dock_layout->addWidget(btnStopProcEarly);
  }

  // Central scroll area
  auto *centralWidget = new QScrollArea(this);
  centralWidget->setWidgetResizable(true);
  this->setCentralWidget(centralWidget);

  auto *layout = new QVBoxLayout(centralWidget);
  centralWidget->setLayout(layout);

  auto *modeSwitchButton = new QPushButton("Switch Mode", this);
  connect(modeSwitchButton, &QPushButton::clicked, this,
          &MainWindow::switchMode);
  layout->addWidget(modeSwitchButton);

  // Add mode views
  // stackedWidget->addWidget(new RealTimeView());
  // stackedWidget->addWidget(new PostProcessingView());

  //   layout->addWidget(modeSwitchButton);
  layout->addWidget(coregDisplay);

  auto pixmap = QPixmap(":/resources/images/radial_380.png");
  coregDisplay->canvas1->imshow(pixmap);
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
  coregDisplay->canvas1->imshow(img1);
  coregDisplay->canvas2->imshow(img2);
}