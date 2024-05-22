#pragma once

#include <QGroupBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QString>

class FrameController : public QGroupBox {
  Q_OBJECT
public:
  explicit FrameController(QWidget *parent = nullptr);

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