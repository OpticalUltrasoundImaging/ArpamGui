#pragma once

#include "Common.hpp"
#include "RFBuffer.hpp"
#include <QDebug>
#include <QtLogging>
#include <filesystem>
#include <memory>
#include <uspam/io.hpp>
#include <uspam/ioParams.hpp>

// Virtual base class defines the interface for a RF producer
class RFProducer {
public:
  explicit RFProducer(const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer)
      : m_buffer(buffer) {}
  RFProducer(const RFProducer &) = delete;
  RFProducer(RFProducer &&) = delete;
  RFProducer &operator=(const RFProducer &) = delete;
  RFProducer &operator=(RFProducer &&) = delete;
  virtual ~RFProducer();

  [[nodiscard]] auto buffer() const { return m_buffer; }

  // start produce frames sequentially. This should be started in a separate
  // thread
  virtual void beginProduce();

  // Generate one frame. For file producer, generate frame at idx
  void produceOne(int idx);

  // Stop producing
  void stop();

  // Release any hardware/file resources
  void release();

  virtual void setIOParams(const uspam::io::IOParams &ioparams);

private:
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;
};

namespace fs = std::filesystem;

class RFProducerFile : public RFProducer {
public:
  explicit RFProducerFile(const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer)
      : RFProducer(buffer), m_ioparams(uspam::io::IOParams::system2024v1()) {}
  void resetParams() { m_ioparams = uspam::io::IOParams::system2024v1(); }

  void setBinpath(const fs::path &binfile) {
    m_binpath = binfile;

    try {
      // Init loader
      m_loader.setParams(m_ioparams);
      m_loader.open(m_binpath);
    } catch (const std::runtime_error &e) {
      const auto msg = QString("RFProducerFile exception: ") +
                       QString::fromStdString(e.what());
      qWarning() << msg;
    }
  }
  auto binpath() const { return m_binpath; }

  void beginProduce() override {}

  void setIOParams(const uspam::io::IOParams &ioparams) override {
    // IOParams is trivially copiable so std::move === memcpy
    m_ioparams = ioparams;
  }

private:
  uspam::io::IOParams m_ioparams;

  fs::path m_binpath;
  uspam::io::BinfileLoader<uint16_t> m_loader;
};