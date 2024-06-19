#pragma once

#include "CoregDisplay.hpp"
#include "DataProcWorker.hpp"
#include <Annotation/AnnotationJsonFile.hpp>
#include <QAction>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QString>
#include <rapidjson/document.h>

class FrameController : public QWidget {
  Q_OBJECT
public:
  // FrameController does not own the worker or the coregDisplay (both are owned
  // by MainWindow). It merely keeps a reference to it for control
  explicit FrameController(DataProcWorker *worker, CoregDisplay *coregDisplay,
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

signals:
  void sigBinfileSelected(QString);
  void sigPlay();
  void sigPause();
  void sigFrameNumUpdated(int);

public:
  auto get_actOpenFileSelectDialog() { return m_actOpenFileSelectDialog; }

private:
  DataProcWorker *m_worker;
  CoregDisplay *m_coregDisplay;

  QPushButton *m_btnPlayPause;
  QAction *m_actOpenFileSelectDialog;

  QSpinBox *m_frameNumSpinBox;
  QSlider *m_frameSlider;
  bool m_isPlaying{false};

  // Annotation JSON document
  annotation::AnnotationJsonFile m_doc;

  // Binfile path
  fs::path m_binPath;
  fs::path m_annoPath;
};
