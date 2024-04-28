#include "MainWindow.hpp"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  auto *centralWidget = new QWidget(this);
  setCentralWidget(centralWidget);

  auto *layout = new QVBoxLayout(centralWidget);
  auto *modeSwitchButton = new QPushButton("Switch Mode", this);
  connect(modeSwitchButton, &QPushButton::clicked, this,
          &MainWindow::switchMode);

  // Stacked widget to switch between different modes
  stackedWidget = new QStackedWidget(this);
  layout->addWidget(stackedWidget);

  // Add mode views
  // stackedWidget->addWidget(new RealTimeView());
  // stackedWidget->addWidget(new PostProcessingView());

  layout->addWidget(modeSwitchButton);
}

void MainWindow::switchMode() {
  int currentIndex = stackedWidget->currentIndex();
  stackedWidget->setCurrentIndex(1 - currentIndex); // Toggle between 0 and 1
}