#pragma once

#include <QAction>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QStackedWidget>
#include <QThread>
#include <QVBoxLayout>
#include <qevent.h>

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

  void handleNewImages(QImage img1, QImage img2, double pix2m);

protected:
  // Support dropping file
  void dragEnterEvent(QDragEnterEvent *event) override;
  void dropEvent(QDropEvent *event) override;

private:
  void closeEvent(QCloseEvent *event) override;

  QThread workerThread;
  DataProcWorker *worker;

  QPlainTextEdit *textEdit;
  // CoregDisplay *coregDisplay;
  Canvas *canvasLeft;
  Canvas *canvasRight;

  FrameController *m_frameController;

  // Actions

  // Action to undo the last cursor action in the canvas
  QAction *actCursorUndo;
  // Action to set the cursor mode to line measure
  QAction *actCursorLine;
  // Action to set the cursor mode to box zoom
  QAction *actCursorZoom;

  // Action to set the cursor mode to box annotation
  QAction *actCursorAnnoBox;
  // Action to set the cursor mode to fan annotation
  QAction *actCursorAnnoFan;
};