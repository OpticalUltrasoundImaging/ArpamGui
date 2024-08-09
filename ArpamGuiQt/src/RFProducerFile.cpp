#include "RFProducerFile.hpp"
#include <QDebug>
#include <QString>
#include <QtLogging>

void RFProducerFile::setBinfile(const fs::path &binfile) {
  try {
    // Init loader
    m_loader.setParams(m_ioparams);
    m_loader.open(binfile);
    emit maxFramesChanged(m_loader.size());

    // Load first frame
    produceOne(0);

  } catch (const std::runtime_error &e) {
    const auto msg = QString("RFProducerFile exception: ") +
                     QString::fromStdString(e.what());
    qCritical() << msg;
    emit messageDialog(msg);
  }
}

void RFProducerFile::closeBinfile() {
  stopProducing();
  m_loader.close();
}

void RFProducerFile::beginProducing() {
  assert(m_loader.isOpen());

  m_producing = true;
  while (m_producing && m_loader.idx() + 1 < m_loader.size()) {
    m_loader.setIdx(m_loader.idx() + 1);
    reproduceOne();
  }
  m_producing = false;

  emit finishedProducing();
}

void RFProducerFile::produceOne(int idx) {
  m_loader.setIdx(idx);
  reproduceOne();
}

void RFProducerFile::reproduceOne() {
  m_buffer->produce([this](std::shared_ptr<BScanData<ArpamFloat>> &data) {
    auto &metrics = data->metrics;

    data->frameIdx = m_loader.idx();
    // qDebug() << "RFProducerFile:: produced idx =" << data->frameIdx;

    // Read RF scan at idx from file
    {
      const uspam::TimeIt timeit;
      m_loader.get<ArpamFloat>(data->rf);
      metrics.load_ms = timeit.get_ms();
    }
  });
};

void RFProducerFile::setIOParams(const uspam::io::IOParams &ioparams) {
  std::unique_lock<std::mutex> lock(m_paramsMtx);
  // IOParams is trivially copiable so std::move === memcpy
  m_ioparams = ioparams;
}
