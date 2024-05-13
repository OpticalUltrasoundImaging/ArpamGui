#pragma once

#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QString>
#include <QWidget>

class FrameController : public QWidget {
  Q_OBJECT
public:
  explicit FrameController(QWidget *parent = nullptr);

private slots:
  void s_openBinFile();

public slots:
  void updateFrameNum(int frameNum);
  void updateMaxFrameNum(int maxFrameNum);

  void updatePlayingState(bool playing);
  void updatePlayingStatePlay();
  void updatePlayingStatePause();

signals:
  void binfileSelected(QString);
  void playClicked();
  void pauseClicked();
  void frameNumUpdated(int);

private:
  QPushButton *btnPlay;
  QPushButton *btnPause;

  QSpinBox *frameNumSpinBox;
  QSlider *frameSlider;
};