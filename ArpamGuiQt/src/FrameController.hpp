#pragma once

#include "AScanPlot.hpp"
#include "Common.hpp"
#include "CoregDisplay.hpp"
#include "RFProducerFile.hpp"
#include "ReconParamsController.hpp"
#include "ReconWorker.hpp"
#include <Annotation/AnnotationJsonFile.hpp>
#include <QAction>
#include <QMenu>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QString>
#include <memory>
#include <rapidjson/document.h>

class FrameController : public QWidget {
  Q_OBJECT
public:
  // FrameController does not own the worker or the coregDisplay (both are owned
  // by MainWindow). It merely keeps a reference to it for control
  explicit FrameController(RFProducerFile *rfProducerFile,
                           ReconWorker *reconWorker,
                           ReconParamsController *paramsController,
                           AScanPlot *ascanPlot, CoregDisplay *coregDisplay,
                           QWidget *parent = nullptr);

  [[nodiscard]] auto frameMenu() const { return m_menu; }

public slots:
  // Open file select dialog
  void openFileSelectDialog();

  // Accept a binfile
  void acceptBinfile(const QString &filename);
  void closeBinfile();

  void receiveNewFrame(std::shared_ptr<BScanData<ArpamFloat>> data);

  [[nodiscard]] int frameNum() const { return m_frameSlider->value(); }
  void setFrameNum(int frameNum);

  [[nodiscard]] int maxFrameNum() const;
  void setMaxFrameNum(int maxFrameNum);

  void updatePlayingState(bool playing);
  void togglePlayPause();

  void nextFrame();
  void prevFrame();

  void plotCurrentBScan();

  // Export frame buffer to a new folder
  void exportCurrentFrame(const fs::path &exportDir);

  void exportAllFrames();

  void handleExportAllFramesBtnClick();

signals:
  void message(QString);
  void statusMessage(QString message, int timeout = 0);
  void sigBinfileSelected(QString);
  void sigFrameNumUpdated(int);

public:
  auto actOpenFileSelectDialog() { return m_actOpenFileSelectDialog; }
  auto actCloseBinfile() { return m_actCloseBinfile; }

private:
  // Annotations
  void saveCurrAnnotationAndLoadNewFrame(int newFrame);
  // Write the current annotation to the given frame number in the JSON doc
  bool saveFrameAnnotationsFromModelToDoc(int frame);
  // Load annotations from the given frame number from the JSON doc
  void loadFrameAnnotationsFromDocToModel(int frame);

  // RF producers
  RFProducerFile *m_producerFile;

  // RF consumer
  ReconWorker *m_reconWorker;

  // Params controller
  ReconParamsController *m_reconParams;

  // Ptr to the coregDisplay for showing a pair of BScan images
  CoregDisplay *m_coregDisplay;

  // Ptr to the AScanPlot to control 2D plotting
  AScanPlot *m_AScanPlot;

  // UI elements
  QPushButton *m_btnPlayPause;
  QSlider *m_frameSlider;
  QPushButton *m_btnExportFrame;
  QPushButton *m_btnExportAllFrames;

  // Actions
  QMenu *m_menu;
  QAction *m_actOpenFileSelectDialog;
  QAction *m_actCloseBinfile;

  QAction *m_actPlayPause;
  QAction *m_actNextFrame;
  QAction *m_actPrevFrame;
  QAction *m_actExportFrame;
  QAction *m_actExportAllFrames;

  // Bscan Data. Processing is done in the worker, and a pointer of the current
  // result is stored here
  std::shared_ptr<BScanData<ArpamFloat>> m_data;

  // Annotation JSON document
  annotation::AnnotationJsonFile m_doc;

  // Binfile path
  fs::path m_binPath;
  fs::path m_annoPath;

  // States
  bool m_isPlaying{false};
  bool m_exportingAllFrames{false};
  fs::path m_exportDir;        // Only used when exporting all
  fs::path m_exportDirDefault; // Desktop
  QString m_sequenceName;
};
