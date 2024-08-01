#pragma once

#include "Common.hpp"
#include "DAQ/DAQ.hpp"
#include <QPushButton>
#include <QThread>
#include <QWidget>
#include <RFBuffer.hpp>
#include <memory>

/*
This widget acts as the data acquisition control UI

*/
class AcquisitionController : public QWidget {
  Q_OBJECT
public:
  explicit AcquisitionController(
      const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer);
  AcquisitionController(const AcquisitionController &) = delete;
  AcquisitionController(AcquisitionController &&) = delete;
  AcquisitionController &operator=(const AcquisitionController &) = delete;
  AcquisitionController &operator=(AcquisitionController &&) = delete;
  ~AcquisitionController() override;

private:
  // Buffer
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  // DAQ
  daq::DAQ *m_daq;
  QThread m_daqThread;

  // UI
  QPushButton *m_btnInitBoard;
  QPushButton *m_btnStartStopAcquisition;
};