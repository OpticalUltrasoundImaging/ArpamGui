#pragma once

#include <QImage>
#include <QMutex>
#include <QThread>
#include <QWaitCondition>
#include <uspam/io.hpp>
#include <uspam/recon.hpp>
#include <uspam/uspam.hpp>

class DataProcessingThread : public QThread {
  Q_OBJECT

public:
  DataProcessingThread() {
    // ioparams = uspam::io::IOParams::system2024v1();
    // params = uspam::recon::ReconParams2::system2024v1();
  }

  // static void setReconParams(const uspam::recon::ReconParams2 &p) {
  //   params = p;
  // }
  // static void setIOParams(const uspam::io::IOParams &p) { ioparams = p; }

public slots:
  void setBinfile(const QString &binfile);
  void stopCurentWork();
  void threadShouldStop();

protected:
  void run() override;

signals:
  void resultReady(QImage img1, QImage img2);
  void finishedOneFile();
  void error(QString err);

private:
  void processCurrentBinfile();

  bool _shouldStopThread{false};
  bool _abortCurrent{false};
  bool _ready{false};
  QMutex _mutex;
  QWaitCondition _condition;

  QString currentBinfile;

  // static uspam::recon::ReconParams2 params;
  // static uspam::io::IOParams ioparams;
};