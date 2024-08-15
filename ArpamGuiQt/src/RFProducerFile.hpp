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
  explicit RFProducerFile(std::shared_ptr<RFBuffer<ArpamFloat>> buffer,
                          const uspam::io::IOParams &ioparams)
      : m_buffer(std::move(buffer)) {
    m_loader.setParams(ioparams);
  }

  bool ready() { return m_loader.isOpen(); }

  void setBinfile(const fs::path &binfile);
  void closeBinfile();

  void beginProducing();
  bool producing() const { return m_producing; }
  void produceOne(int idx);
  void reproduceOne();
  void stopProducing() { m_producing = false; }

  auto size() const { return m_loader.size(); }

  void setIOParams(const uspam::io::IOParams &ioparams);

signals:
  void maxFramesChanged(int);
  void finishedProducing();
  void messageDialog(QString msg);

private:
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  // IO Params. Can be concurrently accessed
  std::mutex m_paramsMtx;
  // File loader
  uspam::io::BinfileLoader<uint16_t> m_loader;

  // Producing state
  std::atomic<bool> m_producing{false};
};