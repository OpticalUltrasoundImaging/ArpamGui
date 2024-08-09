#pragma once

#include "AScanPlot.hpp"
#include "Common.hpp"
#include "CoregDisplay.hpp"
#include "FrameController.hpp"
#include "RFBuffer.hpp"
#include "RFProducerFile.hpp"
#include "ReconWorker.hpp"
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
#include <memory>
#include <qdockwidget.h>
#include <qevent.h>

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = nullptr);
  MainWindow(const MainWindow &) = delete;
  MainWindow(MainWindow &&) = delete;
  MainWindow &operator=(const MainWindow &) = delete;
  MainWindow &operator=(MainWindow &&) = delete;
  ~MainWindow() override;

public slots:
  // Log error message to a text box on screen
  void logError(QString message);
  void messageBox(const QString &title, const QString &message);

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
  QDockWidget *dockAcquisitionController;

  // Actions
  QAction *actViewSimple;
  QAction *actViewExpert;

  // Shared thread safe Buffer
  std::shared_ptr<RFBuffer<ArpamFloat>> buffer;

  // File loader (producer)
  QThread producerThreadFile;
  RFProducerFile *rfProducerFile;

  // Recon Worker (consumer)
  QThread workerThread;
  ReconWorker *reconWorker;

  QPlainTextEdit *textEdit;
  ReconParamsController *reconParamsController;

  // AScan plot
  AScanPlot *m_AScanPlot;
  // BScan display
  CoregDisplay *m_coregDisplay;
  // Controller
  FrameController *m_frameController;
};
