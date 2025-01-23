#include "RFProducerFile.hpp"
#include <QDebug>
#include <QString>
#include <QtLogging>
#include <regex>

void RFProducerFile::setBinfile(const fs::path &binfile) {
  try {
    // Init loader
    m_loader.open(binfile);
    emit maxFramesChanged(m_loader.size());

    // Parse scansEachDirection
    {
      const auto stem = binfile.stem().string();
      std::regex regex(R"rgx(_SED(\d+))rgx");
      std::smatch match;
      if (std::regex_search(stem, match, regex)) {
        m_scansEachDirection = std::stoi(match[1].str());
      } else {
        m_scansEachDirection = 1;
      }
    }

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

    // Set flip
    { data->flip = (data->frameIdx / m_scansEachDirection) % 2 == 1; }
  });
};

void RFProducerFile::setIOParams(const uspam::io::IOParams &ioparams) {
  std::unique_lock<std::mutex> lock(m_paramsMtx);
  // IOParams is trivially copiable so std::move === memcpy
  m_loader.setParams(ioparams);
}
