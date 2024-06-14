#pragma once

#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
#include "FrameController.hpp"
#include <QAction>
#include <QActionGroup>
#include <QContextMenuEvent>
#include <QEvent>
#include <QImage>
#include <QKeyEvent>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QStackedWidget>
#include <QThread>
#include <QVBoxLayout>

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = nullptr);

public slots:
  // Log error message to a text box on screen
  void logError(QString message);

  void switchMode();

protected:
  // Support dropping file
  void dragEnterEvent(QDragEnterEvent *event) override;
  void dropEvent(QDropEvent *event) override;
  void keyPressEvent(QKeyEvent *event) override;

  void contextMenuEvent(QContextMenuEvent *event) override;

private:
  void closeEvent(QCloseEvent *event) override;

  // Menu items
  QMenu *m_fileMenu;
  QMenu *m_viewMenu;
  QMenu *m_helpMenu;

  // Actions

  // Worker
  QThread workerThread;
  DataProcWorker *worker;

  QPlainTextEdit *textEdit;
  FrameController *m_frameController;
  CoregDisplay *m_coregDisplay;
};
