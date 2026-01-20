#pragma once

#include "Annotation/Annotation.hpp"
#include <QImage>
#include <QList>
#include <QString>
#include <armadillo>
#include <array>
#include <condition_variable>
#include <filesystem>
#include <fmt/core.h>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct PerformanceMetrics {
  float total_ms{};
  float load_ms{};
  float split_ms{};
  float beamform_ms{};
  float recon_ms{};
  float imageConversion_ms{};
  float overlay_ms{};

  // Template function to handle the common formatting
  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const PerformanceMetrics &pm) {
    stream << "total " << static_cast<int>(pm.total_ms) << ", loadRf "
           << static_cast<int>(pm.load_ms) << ", split "
           << static_cast<int>(pm.split_ms) << ", beamform "
           << static_cast<int>(pm.beamform_ms) << ", recon "
           << static_cast<int>(pm.recon_ms) << ", imageConversion "
           << static_cast<int>(pm.imageConversion_ms) << ", overlay "
           << static_cast<int>(pm.overlay_ms);
    return stream;
  }
};

struct ExportSetting {
  bool saveRF{false};
  bool saveRadialImages{true};
  bool saveRectImages{true};
  bool saveRoiOnly{false};
};

template <typename T> struct BScanData_ {
  // Buffers
  arma::Mat<T> rf;
  // arma::Mat<T> rfFilt;
  arma::Mat<T> rfBeamformed;
  arma::Mat<T> rfEnv;
  arma::Mat<uint8_t> rfLog;

  // Images
  cv::Mat radial;
  QImage radial_img;

  void saveBScanData(const fs::path &directory, const std::string &prefix = "",
                     const ExportSetting &exportSetting = {}) const;
};

/*
 * Contains all the data for one BScan
 * From RF to Image
 */
template <typename T> struct BScanData {
  // RF data
  arma::Mat<T> rf;
  bool flip{};

  BScanData_<T> PA;
  BScanData_<T> US;

  cv::Mat PAUSradial; // CV_8U3C
  QImage PAUSradial_img;

  // [mm] spacing of one radial pixel
  double spacingRadialUS{};
  double spacingRectUS{};

  // Frame idx
  int frameIdx{};

  // Metrics
  PerformanceMetrics metrics;

  // Export Bscan data to the directory.
  // directory should be created new for each frame
  void exportToFile(const fs::path &directory,
                    const QList<annotation::Annotation> &annotations = {},
                    const ExportSetting &exportSetting = {}) const;

private:
  // Helper to export annotated crops
  void exportAnnotatedCrops(const fs::path &directory,
                            const QList<annotation::Annotation> &annotations,
                            const ExportSetting &exportSetting) const;
};

/*
Thread safe buffer for the producer/consumer pattern

Inspired by https://andrew128.github.io/ProducerConsumer/
*/
template <typename T> class RFBuffer {
public:
  RFBuffer() {
    for (auto &sptr : m_buffer) {
      sptr = std::make_shared<BScanData<T>>();
    }
  }

  // The exit condition for a consumer is receiving a nullptr
  void exit() {
    std::unique_lock<std::mutex> unique_lock(m_mtx);

    m_not_full.wait(unique_lock,
                    [this] { return buffer_size != m_buffer.size(); });

    m_buffer[m_wpos] = nullptr;

    // Update fields
    m_wpos = (m_wpos + 1) % m_buffer.size();
    buffer_size++;

    unique_lock.unlock();
    // Notify one thread that buffer isn't empty
    m_not_empty.notify_one();
  }

  template <typename Func> void produce(const Func &producer_callback) {
    std::unique_lock<std::mutex> unique_lock(m_mtx);

    // Wait if the buffer is full
    m_not_full.wait(unique_lock,
                    [this] { return buffer_size != m_buffer.size(); });

    // Add to buffer
    producer_callback(m_buffer[m_wpos]);

    // Update fields
    m_wpos = (m_wpos + 1) % m_buffer.size();
    buffer_size++;

    unique_lock.unlock();
    // Notify one thread that buffer isn't empty
    m_not_empty.notify_one();
  }

  template <typename Func> void consume(const Func &consumer_callback) {
    // Acquire
    std::unique_lock<std::mutex> unique_lock(m_mtx);

    // Wait if buffer is empty
    m_not_empty.wait(unique_lock, [this]() { return buffer_size != 0; });

    // Get from buffer
    consumer_callback(m_buffer[m_rpos]);

    // Update appropriate fields
    m_rpos = (m_rpos + 1) % m_buffer.size();
    buffer_size--;

    unique_lock.unlock();
    // Notify one thread that the buffer isn't full
    m_not_full.notify_one();
  }

private:
  std::array<std::shared_ptr<BScanData<T>>, 10> m_buffer{};
  int buffer_size{};
  int m_rpos{}; // read idx where vars are removed from buffer (consumed)
  int m_wpos{}; // write idx where vars are put inside of buffer (produced)

  // Concurrency
  std::mutex m_mtx;
  std::condition_variable m_not_empty;
  std::condition_variable m_not_full;
};
