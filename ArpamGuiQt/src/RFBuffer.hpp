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
#include <uspam/uspam.hpp>

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

template <uspam::Floating T> struct BScanData_ {
  // Buffers
  arma::Mat<T> rf;
  // arma::Mat<T> rfFilt;
  arma::Mat<T> rfBeamformed;
  arma::Mat<T> rfEnv;
  arma::Mat<uint8_t> rfLog;

  // Images
  cv::Mat radial;
  QImage radial_img;

  void saveBScanData(const fs::path &directory, const std::string &prefix = "");
};

/*
 * Contains all the data for one BScan
 * From RF to Image
 */
template <uspam::Floating T> struct BScanData {
  // RF data
  arma::Mat<T> rf;

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
                    const QList<annotation::Annotation> &annotations = {});
};

/*
Thread safe buffer for the producer/consumer pattern

Inspired by https://andrew128.github.io/ProducerConsumer/
*/
template <uspam::Floating T> class RFBuffer {
public:
  RFBuffer() {
    for (int i = 0; i < buffer.size(); ++i) {
      buffer[i] = std::make_shared<BScanData<T>>();
    }
  }

  // The exit condition for a consumer is receiving a nullptr
  void exit() {
    std::unique_lock<std::mutex> unique_lock(mtx);

    not_full.wait(unique_lock, [this] { return buffer_size != buffer.size(); });

    buffer[right] = nullptr;

    // Update fields
    right = (right + 1) % buffer.size();
    buffer_size++;

    // Unlock unique lock
    unique_lock.unlock();
    // Notify one thread that buffer isn't empty
    not_empty.notify_one();
  }

  template <typename Func> void produce(const Func &producer_callback) {
    std::unique_lock<std::mutex> unique_lock(mtx);

    // Wait if the buffer is full
    not_full.wait(unique_lock, [this] { return buffer_size != buffer.size(); });

    // Add input to buffer
    // buffer[right] = num;
    auto &currData = buffer[right];
    { producer_callback(currData); }

    // Update fields
    right = (right + 1) % buffer.size();
    buffer_size++;

    // Unlock unique lock
    unique_lock.unlock();
    // Notify one thread that buffer isn't empty
    not_empty.notify_one();
  }

  template <typename Func> void consume(const Func &consumer_callback) {
    // Acquire
    std::unique_lock<std::mutex> unique_lock(mtx);

    // Wait if buffer is empty
    not_empty.wait(unique_lock, [this]() { return buffer_size != 0; });

    // Get from buffer
    {
      auto &result = buffer[left];
      consumer_callback(result);
    }

    // Update appropriate fields
    left = (left + 1) % buffer.size();
    buffer_size--;

    // Unlock unique lock
    unique_lock.unlock();
    // Notify one thread that the buffer isn't full
    not_full.notify_one();
  }

private:
  std::array<std::shared_ptr<BScanData<T>>, 4> buffer{};
  int buffer_size{0};
  int left{0};  // index where vars are put inside of buffer (produced)
  int right{0}; // idx where vars are removed from buffer (consumed)

  // Concurrency
  std::mutex mtx;
  std::condition_variable not_empty;
  std::condition_variable not_full;
};