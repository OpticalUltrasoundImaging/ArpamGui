#pragma once

#include <QImage>
#include <QMutex>
#include <QMutexLocker>
#include <QObject>
#include <atomic>
#include <filesystem>
#include <uspam/io.hpp>
#include <uspam/recon.hpp>
#include <uspam/uspam.hpp>

namespace fs = std::filesystem;

class DataProcWorker : public QObject {
  Q_OBJECT

public:
  DataProcWorker() {
    ioparams = uspam::io::IOParams::system2024v1();
    params = uspam::recon::ReconParams2::system2024v1();
  }

  // static void setReconParams(const uspam::recon::ReconParams2 &p) {
  //   params = p;
  // }
  // static void setIOParams(const uspam::io::IOParams &p) { ioparams = p; }

public slots:
  void setBinfile(const QString &binfile);

  // Begin post processing data using the currentBinfile
  void doPostProcess();

  // Returns true if the worker is ready to start new work.
  inline bool isReady() { return _ready; };

  // Abort the current work (only works when ready=false. Updates ready=true)
  void abortCurrentWork();

  // This slot must be called in the calling thread (not in the worker thread)
  inline void updateParams(uspam::recon::ReconParams2 params,
                           uspam::io::IOParams ioparams) {

    emit error("DataProcWorker updateParams");
    QMutexLocker lock(&paramsMutex);
    this->params = std::move(params);
    this->ioparams = std::move(ioparams);
  }

signals:
  void updateMaxFrames(int);
  void updateFrameIdx(int);

  void resultReady(QImage img1, QImage img2);
  void finishedOneFile();
  void error(QString err);

private:
  void processCurrentBinfile();
  void processCurrentFrame();

private:
  int frameIdx{0};

  // Post processing binfile
  uspam::io::BinfileLoader<uint16_t> loader;
  fs::path binfilePath;
  fs::path imageSaveDir;

  // Buffers;
  arma::Mat<uint16_t> rf;
  uspam::io::PAUSpair<double> rfPair;
  uspam::io::PAUSpair<uint8_t> rfLog;

  // Atomic states
  std::atomic<bool> _abortCurrent{false};
  std::atomic<bool> _ready{true};

  // mutex for ReconParams2 and IOParams
  QMutex paramsMutex;

  uspam::recon::ReconParams2 params;
  uspam::io::IOParams ioparams;
};