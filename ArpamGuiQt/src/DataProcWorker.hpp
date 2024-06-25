#pragma once

#include <QImage>
#include <QMutex>
#include <QMutexLocker>
#include <QObject>
#include <QWaitCondition>
#include <atomic>
#include <filesystem>
#include <memory>
#include <uspam/io.hpp>
#include <uspam/recon.hpp>
#include <uspam/uspam.hpp>

namespace fs = std::filesystem;

struct PerformanceMetrics {
  float fileloader_ms{};
  float splitRfPAUS_ms{};
  float reconUSPA_ms{};
  float makeOverlay_ms{};
  float writeImages_ms{};

  // Template function to handle the common formatting
  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const PerformanceMetrics &pm) {
    stream << "fileloader " << static_cast<int>(pm.fileloader_ms) << ", "
           << "splitRfPAUS " << static_cast<int>(pm.splitRfPAUS_ms) << ", "
           << "reconUSPA " << static_cast<int>(pm.reconUSPA_ms) << ", "
           << "makeOverlay " << static_cast<int>(pm.makeOverlay_ms) << ", "
           << "writeImages " << static_cast<int>(pm.writeImages_ms);
    return stream;
  }
};

/**
 * Contains all the data for one BScan
 * From RF to Image
 *
 * For initialization, only PAUSpair need to be explicitly allocated since
 * `rf` will be overwritten, and cv::Mat and QImage have default constructors
 */
template <uspam::Floating FloatType> struct BScanData {
  // RF data
  arma::Mat<FloatType> rf;
  uspam::io::PAUSpair<FloatType> rfPair;
  uspam::io::PAUSpair<uint8_t> rfLog;

  // Images
  cv::Mat USradial;   // CV_8U1C
  cv::Mat PAradial;   // CV_8U1C
  cv::Mat PAUSradial; // CV_8U3C

  QImage USradial_img;
  QImage PAradial_img;
  QImage PAUSradial_img;

  // depth [m] of one radial pixel
  double fct{};

  // Frame idx
  int frameIdx{};

  BScanData(const uspam::io::IOParams &ioparams, int alinesPerBscan)
      : rfPair(ioparams.allocateSplitPair<FloatType>(alinesPerBscan)),
        rfLog(uspam::io::PAUSpair<uint8_t>::empty_like(rfPair)) {}
};

/**
 * Data processing worker that will be launched in a QThread
 */
class DataProcWorker : public QObject {
  Q_OBJECT

public:
  using FloatType = float;

  DataProcWorker()
      : m_params(uspam::recon::ReconParams2::system2024v1()),
        m_ioparams(uspam::io::IOParams::system2024v1()) {}

  // Returns true if the worker is currently playing (sequentially processing)
  inline bool isPlaying() { return m_isPlaying; }

  // Returns true if the worker has a binfile ready to process
  inline bool isReady() { return m_ready; }

public slots:
  // Begin post processing data using the currentBinfile
  void setBinfile(const fs::path &binfile);

  // Start processing frames sequentially
  // By default start playing at current frameIdx
  void play();
  // Process frame at idx.
  void playOne(int idx);
  // Replay the current frame (without advancing the index)
  void replayOne();

  // (thread safe) If .play() called, pause. This needs to be called in the
  // caller thread Abort the current work
  void pause();

  // Updates the ReconParams and IOParams used for processing
  // This slot must be called in the calling thread (not in the worker thread)
  void updateParams(uspam::recon::ReconParams2 params,
                    uspam::io::IOParams ioparams);

  // Reset the ReconParams and IOParams to the default
  void resetParams() {
    m_ioparams = uspam::io::IOParams::system2024v1();
    m_params = uspam::recon::ReconParams2::system2024v1();
  }

  // Save the ReconParams and IOParams to the image output directory
  void saveParamsToFile();

  inline auto getBinfilePath() const -> fs::path { return this->m_binfilePath; }
  inline auto getImageSaveDir() const -> fs::path {
    return this->m_imageSaveDir;
  }

  void initDataBuffers();

signals:
  void maxFramesChanged(int);
  void frameIdxChanged(int);

  // pix2m is the depth [m] of each radial pixel
  void resultReady(std::shared_ptr<BScanData<FloatType>>);

  void finishedPlaying();
  void error(QString err);

private:
  void processCurrentFrame();

  int m_frameIdx{0};
  std::atomic<bool> m_ready{false};
  std::atomic<bool> m_isPlaying{false};

  // Post processing binfile
  uspam::io::BinfileLoader<uint16_t> m_loader;
  fs::path m_binfilePath;
  fs::path m_imageSaveDir;

  // Buffers;
  std::shared_ptr<BScanData<FloatType>> m_data;

  // mutex for ReconParams2 and IOParams
  QMutex m_paramsMutex;
  QWaitCondition m_waitCondition;

  uspam::recon::ReconParams2 m_params;
  uspam::io::IOParams m_ioparams;
};
