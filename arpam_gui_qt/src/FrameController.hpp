#pragma once

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

signals:
  void openBinFile(QString path);
  void play();
  void pause();
  void frameNumUpdated(int frameNum);

private:
  QSpinBox *frameNumSpinBox;
  QSlider *frameSlider;
};