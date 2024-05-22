#pragma once

#include <QImage>
#include <QMutex>
#include <QMutexLocker>
#include <QObject>
#include <QWaitCondition>
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

  // Returns true if the worker is currently playing (sequentially processing)
  inline bool isPlaying() { return _isPlaying; }

public slots:
  // Begin post processing data using the currentBinfile
  void setBinfile(const QString &binfile);

  // Start processing frames sequentially
  // By default start playing at current frameIdx
  void play();
  // Process frame at idx.
  void playOne(int idx);
  void replayOne();

  // If .play() called, pause. This needs to be called in the caller thread
  // Abort the current work (only works when ready=false. Updates ready=true)
  void pause();

  // This slot must be called in the calling thread (not in the worker thread)
  inline void updateParams(uspam::recon::ReconParams2 params,
                           uspam::io::IOParams ioparams) {

    emit error("DataProcWorker updateParams");
    QMutexLocker lock(&paramsMutex);
    this->params = std::move(params);
    this->ioparams = std::move(ioparams);
  }

  void saveParamsToFile() {
    QMutexLocker lock(&paramsMutex);
    const auto savedir = imageSaveDir;
    params.serializeToFile(savedir / "params.json");
    ioparams.serializeToFile(savedir / "ioparams.json");
  }

  inline auto getBinfilePath() const -> fs::path { return this->binfilePath; }
  inline auto getImageSaveDir() const -> fs::path { return this->imageSaveDir; }

signals:
  void updateMaxFrames(int);
  void updateFrameIdx(int);

  // pix2m is the depth [m] of each radial pixel
  void resultReady(QImage img1, QImage img2, double pix2m);

  void finishedOneFile();
  void error(QString err);

private:
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
  std::atomic<bool> _isPlaying{false};

  // mutex for ReconParams2 and IOParams
  QMutex paramsMutex;
  QWaitCondition waitCondition;

  uspam::recon::ReconParams2 params;
  uspam::io::IOParams ioparams;
};