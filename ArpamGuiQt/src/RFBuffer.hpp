#pragma once

#include <QImage>
#include <armadillo>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <uspam/uspam.hpp>

constexpr int BUFFER_CAPACITY = 4;

template <uspam::Floating T> struct BScanData_ {
  arma::Mat<T> rf;
  arma::Mat<T> rfFilt;
  arma::Mat<T> rfBeamformed;
  arma::Mat<T> rfEnv;
  arma::Mat<uint8_t> rfLog;

  // Images
  cv::Mat radial;
  QImage radial_img;
};

/*
 * Contains all the data for one BScan
 * From RF to Image
 *
 * For initialization, only PAUSpair need to be explicitly allocated since
 * `rf` will be overwritten, and cv::Mat and QImage have default constructors
 */
template <uspam::Floating T> struct BScanData {
  // RF data
  arma::Mat<T> rf;

  BScanData_<T> PA;
  BScanData_<T> US;

  cv::Mat PAUSradial; // CV_8U3C
  QImage PAUSradial_img;

  // depth [m] of one radial pixel
  double fct{};

  // Frame idx
  int frameIdx{};
};

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
  std::array<std::shared_ptr<BScanData<T>>, BUFFER_CAPACITY> buffer{};
  int buffer_size;
  int left;  // index where vars are put inside of buffer (produced)
  int right; // idx where vars are removed from buffer (consumed)

  // Concurrency
  std::mutex mtx;
  std::condition_variable not_empty;
  std::condition_variable not_full;
};