#include "MainWindow.hpp"
#include <QApplication>
#include <QMainWindow>
#include <QtWidgets>
#include <ctime>
#include <format>
#include <fstream>

void addDatetime(std::ostream &os) {
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  os << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";
}

// Log handler
void myMessageHandler(QtMsgType type, const QMessageLogContext &context,
                      const QString &msg) {
  static std::ofstream sLogStream = []() {
    std::ofstream os("arpam-gui.log", std::ios::app);
    os << "\n";
    addDatetime(os);
    os << "Application launched\n";
    return os;
  }();

  const QByteArray localMsg = msg.toLocal8Bit();
  const char *file = context.file != nullptr ? context.file : "";
  const char *function = context.function != nullptr ? context.function : "";

  addDatetime(sLogStream);
  switch (type) {
  case QtDebugMsg:
    sLogStream << "Debug: " << localMsg.constData() << "(" << file << ":"
               << context.line << ")\n";
    break;
  case QtInfoMsg:
    sLogStream << "Info: " << localMsg.constData() << " (" << file << ":"
               << context.line << ")\n";
    break;
  case QtWarningMsg:
    sLogStream << "Warning: " << localMsg.constData() << "(" << file << ":"
               << context.line << ")\n";
    break;
  case QtCriticalMsg:
    sLogStream << "Critical: " << localMsg.constData() << "(" << file << ":"
               << context.line << ")\n";
    break;
  case QtFatalMsg:
    sLogStream << "Fatal: " << localMsg.constData() << "(" << file << ":"
               << context.line << ")\n";
    break;
  }
}

auto main(int argc, char **argv) -> int {
  qInstallMessageHandler(myMessageHandler);

  QApplication app(argc, argv);
  MainWindow mainWindow;
  mainWindow.showMaximized();
  return app.exec();
}