#pragma once

#include "Common.hpp"
#include "RFBuffer.hpp"
#include "Recon.hpp"
#include <QObject>
#include <atomic>
#include <utility>

class ReconWorker : public QObject {
  Q_OBJECT
public:
  explicit ReconWorker(std::shared_ptr<RFBuffer<ArpamFloat>> buffer)
      : m_buffer(std::move(buffer)) {}

  auto &reconstructor() { return m_recontsructor; }
  auto &reconstructor() const { return m_recontsructor; }

  /*
  Start main loop
  */
  inline auto start() {
    bool m_shouldStop = false;
    while (!m_shouldStop) {
      m_buffer->consume([&](std::shared_ptr<BScanData<ArpamFloat>> &data) {
        if (data == nullptr) {
          m_shouldStop = true;
        } else {
          m_recontsructor.recon(*data);
          emit imagesReady(data);
        }
      });
    }
  }

signals:
  void imagesReady(std::shared_ptr<BScanData<ArpamFloat>> data);

private:
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  Recon::Reconstructor m_recontsructor;
};