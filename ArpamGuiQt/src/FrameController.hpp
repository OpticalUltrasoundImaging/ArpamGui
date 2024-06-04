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
  void acceptNewBinfile(const QString &filename);
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
  QPushButton *m_btnPlay;
  QPushButton *m_btnPause;

  QSpinBox *m_frameNumSpinBox;
  QSlider *m_frameSlider;
};