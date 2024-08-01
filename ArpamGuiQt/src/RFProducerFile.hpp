#pragma once

#include "Common.hpp"
#include "RFBuffer.hpp"
#include <QObject>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <uspam/io.hpp>
#include <uspam/ioParams.hpp>
#include <uspam/timeit.hpp>
#include <utility>

namespace fs = std::filesystem;

class RFProducerFile : public QObject {
  Q_OBJECT
public:
  explicit RFProducerFile(std::shared_ptr<RFBuffer<ArpamFloat>> buffer)
      : m_buffer(std::move(buffer)),
        m_ioparams(uspam::io::IOParams::system2024v1()) {}
  void resetParams() { m_ioparams = uspam::io::IOParams::system2024v1(); }

  bool ready() { return m_loader.isOpen(); }

  void setBinpath(const fs::path &binfile);

  void beginProducing();
  bool producing() const { return m_producing; }
  void produceOne(int idx);
  void reproduceOne();
  void stopProducing() { m_producing = false; }

  void setIOParams(const uspam::io::IOParams &ioparams);

signals:
  void maxFramesChanged(int);
  void finishedProducing();
  void messageDialog(QString msg);

private:
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  // IO Params. Can be concurrently accessed
  std::mutex m_paramsMtx;
  uspam::io::IOParams m_ioparams;

  // File loader
  uspam::io::BinfileLoader<uint16_t> m_loader;

  // Producing state
  std::atomic<bool> m_producing{false};
};