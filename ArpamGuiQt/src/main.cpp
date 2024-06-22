#include "MainWindow.hpp"
#include "datetime.hpp"
#include <QApplication>
#include <QFile>
#include <QMainWindow>
#include <QStyle>
#include <QStyleHints>
#include <QtWidgets>
#include <ctime>
#include <fftconv.hpp>
#include <fftw3.h>
#include <fstream>
#include <uspam/fft.hpp>

void addDatetime(std::ostream &os) {
  os << "[" << datetime::datetimeFormat("%Y-%m-%d %H:%M:%S") << "] ";
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

  sLogStream << localMsg.constData();
  // sLogStream << "(" << file << ":" << context.line;
  sLogStream << ")\n";
}

auto main(int argc, char **argv) -> int {
  // Make FFTW thread safe
  fftw_make_planner_thread_safe();
  fftwf_make_planner_thread_safe();

  // qInstallMessageHandler(myMessageHandler);

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

  return QApplication::exec();
}
