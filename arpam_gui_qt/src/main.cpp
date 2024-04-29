#include "MainWindow.hpp"
#include <QApplication>
#include <QMainWindow>
#include <QtWidgets>
#include <chrono>
#include <ctime>
#include <format>
#include <fstream>

void addDatetime(std::ostream &os) {
  auto const time =
      std::chrono::current_zone()->to_local(std::chrono::system_clock::now());
  os << "[" << std::format("{:%Y-%m-%d %X}", time) << "] ";
}

// Log handler
void myMessageHandler(QtMsgType type, const QMessageLogContext &context,
                      const QString &msg) {
  static std::ofstream sLogStream = []() {
    std::ofstream os("arpam-gui.log", std::ios::app);
    os << "\n";

    const auto appDirPath = QCoreApplication::applicationDirPath();

    addDatetime(os);
    os << "Application launched (" << appDirPath.toStdString() << ")\n";

    return os;
  }();

  const QByteArray localMsg = msg.toLocal8Bit();
  const char *file = context.file != nullptr ? context.file : "";
  const char *function = context.function != nullptr ? context.function : "";

  addDatetime(sLogStream);
  switch (type) {
  case QtDebugMsg:
    sLogStream << "Debug: ";
    break;
  case QtInfoMsg:
    sLogStream << "Info: ";
    break;
  case QtWarningMsg:
    sLogStream << "Warning: ";
    break;
  case QtCriticalMsg:
    sLogStream << "Critical: ";
    break;
  case QtFatalMsg:
    sLogStream << "Fatal: ";
    break;
  }

  sLogStream << localMsg.constData() << "(" << file << ":" << context.line
             << ")\n";
}

auto main(int argc, char **argv) -> int {
  qInstallMessageHandler(myMessageHandler);

  QApplication app(argc, argv);
  MainWindow mainWindow;
  mainWindow.showMaximized();
  return app.exec();
}