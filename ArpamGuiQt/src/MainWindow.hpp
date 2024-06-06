#pragma once

#include <QAction>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QStackedWidget>
#include <QThread>
#include <QVBoxLayout>

#include "Canvas.hpp"
#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
#include "FrameController.hpp"

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow() = default;

public slots:
  // Log error message to a text box on screen
  void logError(QString message);

  void switchMode();

protected:
  // Support dropping file
  void dragEnterEvent(QDragEnterEvent *event) override;
  void dropEvent(QDropEvent *event) override;
  void keyPressEvent(QKeyEvent *event) override;

private:
  void closeEvent(QCloseEvent *event) override;

  QThread workerThread;
  DataProcWorker *worker;

  QPlainTextEdit *textEdit;
  FrameController *m_frameController;
  CoregDisplay *m_coregDisplay;
};