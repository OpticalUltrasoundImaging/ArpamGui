#include "MainWindow.hpp"
#include <QApplication>
#include <QMainWindow>

auto main(int argc, char **argv) -> int {
  QApplication app(argc, argv);
  MainWindow mainWindow;
  mainWindow.show();
  return app.exec();
}