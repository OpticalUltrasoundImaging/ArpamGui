#pragma once

#include "AScanPlot.hpp"
#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
#include "ReconParamsController.hpp"
#include <Annotation/AnnotationJsonFile.hpp>
#include <QAction>
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
  explicit FrameController(ReconParamsController *paramsController,
                           DataProcWorker *worker, AScanPlot *ascanPlot,
                           CoregDisplay *coregDisplay,
                           QWidget *parent = nullptr);

public slots:
  // Open file select dialog
  void openFileSelectDialog();

  // Accept a binfile
  void acceptNewBinfile(const QString &filename);

  [[nodiscard]] int frameNum() const;
  void setFrameNum(int frameNum);

  [[nodiscard]] int maxFrameNum() const;
  void setMaxFrameNum(int maxFrameNum);

  void updatePlayingState(bool playing);
  void togglePlayPause();

  void nextFrame();
  void prevFrame();

  void AScanIdxUpdated(int idx);

  void plotCurrentAScan();
  void plotCurrentBScan();

signals:
  void message(QString);
  void statusMessage(QString message, int timeout = 0);
  void sigBinfileSelected(QString);
  void sigPlay();
  void sigPause();
  void sigFrameNumUpdated(int);

public:
  auto get_actOpenFileSelectDialog() { return m_actOpenFileSelectDialog; }

private:
  void saveFrameAnnotationsFromModelToDoc(int frame);
  void loadFrameAnnotationsFromDocToModel(int frame);

  ReconParamsController *m_reconParams;
  DataProcWorker *m_worker;

  // Ptr to the coregDisplay for showing a pair of BScan images
  CoregDisplay *m_coregDisplay;

  // Ptr to the AScanPlot to control 2D plotting
  AScanPlot *m_AScanPlot;
  int m_AScanPlotIdx_canvas{}; // Received from canvas, not corrected for flip
                               // and rotation
  int m_AScanPlotIdx{};        // Corrected for flip and rotation

  QPushButton *m_btnPlayPause;
  QAction *m_actOpenFileSelectDialog;

  QSpinBox *m_frameNumSpinBox;
  QSlider *m_frameSlider;
  bool m_isPlaying{false};

  // Bscan Data. Processing is done in the worker, and a pointer of the current
  // result is stored here
  std::shared_ptr<BScanData<DataProcWorker::FloatType>> m_data;

  // Annotation JSON document
  annotation::AnnotationJsonFile m_doc;

  // Binfile path
  fs::path m_binPath;
  fs::path m_annoPath;
};
