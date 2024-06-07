#pragma once

#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QString>

class FrameController : public QWidget {
  Q_OBJECT
public:
  explicit FrameController(QWidget *parent = nullptr);

public slots:
  void acceptNewBinfile(const QString &filename);
  void updateFrameNum(int frameNum);
  void updateMaxFrameNum(int maxFrameNum);

  void updatePlayingState(bool playing);
  void togglePlayPause();

  void nextFrame();
  void prevFrame();

signals:
  void sigBinfileSelected(QString);
  void sigPlay();
  void sigPause();
  void sigFrameNumUpdated(int);

private:
  QPushButton *m_btnPlayPause;

  QSpinBox *m_frameNumSpinBox;
  QSlider *m_frameSlider;
  bool m_isPlaying{false};
};