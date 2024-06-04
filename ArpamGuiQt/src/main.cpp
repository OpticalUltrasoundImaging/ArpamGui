#include "MainWindow.hpp"
#include <QApplication>
#include <QFile>
#include <QMainWindow>
#include <QStyle>
#include <QStyleHints>
#include <QtWidgets>
#include <ctime>
#include <format>
#include <fstream>

#include <fftconv.hpp>
#include <fftw3.h>
#include <mutex>
#include <uspam/fft.hpp>

// void addDatetime(std::ostream &os) {
//   std::time_t t = std::time(nullptr);
//   std::tm tm = *std::localtime(&t);
//   os << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";
// }

void addDatetime(std::ostream &os) {
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm tm;
#if defined(_WIN32) || defined(_WIN64)
  localtime_s(&tm, &now_time_t); // Use localtime_s on Windows
#else
  localtime_r(&now_time_t, &tm); // Use localtime_r on other platforms
#endif
  os << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";
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
  // const char *function = context.function != nullptr ? context.function : "";

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
  // Make FFTW thread safe
  fftw_make_planner_thread_safe();

  qInstallMessageHandler(myMessageHandler);

  QApplication::setStyle("Fusion"); // Dark mode
  QApplication app(argc, argv);

  // load style sheet
  {
    QFile file(":/resources/styles/styles.qss");
    if (file.open(QFile::ReadOnly)) {
      QString styleSheet = file.readAll();
      app.setStyleSheet(styleSheet);
    }
  }

  MainWindow mainWindow;
  mainWindow.setWindowTitle("ArpamGui");
  mainWindow.showMaximized();

  return app.exec();
}
