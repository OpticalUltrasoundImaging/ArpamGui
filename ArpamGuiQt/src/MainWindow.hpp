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

#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
#include "ImshowCanvas.hpp"

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow() = default;

public slots:
  // Log error message to a text box on screen
  void logError(const QString &message);

  void switchMode();

  void handleNewImages(QImage img1, QImage img2, double pix2m);

private:
  void closeEvent(QCloseEvent *event) override;

  QPlainTextEdit *textEdit;
  // CoregDisplay *coregDisplay;
  ImshowCanvas *canvasLeft;
  ImshowCanvas *canvasRight;

  QThread workerThread;
  DataProcWorker *worker;

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