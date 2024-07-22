#pragma once

#include "AScanPlot.hpp"
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
#include <qdockwidget.h>

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = nullptr);

public slots:
  // Log error message to a text box on screen
  void logError(QString message);

protected:
  // Support dropping file
  void dragEnterEvent(QDragEnterEvent *event) override;
  void dropEvent(QDropEvent *event) override;

private:
  void closeEvent(QCloseEvent *event) override;

  // Menu items
  QMenu *m_fileMenu;
  QMenu *m_viewMenu;

  // Docks
  QDockWidget *dockLog;
  QDockWidget *dockFrameController;
  QDockWidget *dockReconParams;
  QDockWidget *dockAnnotations;
  QDockWidget *dockAScanPlot;

  // Actions
  QAction *actViewSimple;
  QAction *actViewExpert;

  // Worker
  QThread workerThread;
  DataProcWorker *worker;

  QPlainTextEdit *textEdit;
  ReconParamsController *reconParamsController;

  // AScan plot
  AScanPlot *m_AScanPlot;
  // BScan display
  CoregDisplay *m_coregDisplay;
  // Controller
  FrameController *m_frameController;
};
