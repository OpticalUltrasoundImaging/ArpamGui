#pragma once

#include "Common.hpp"
#include "DAQ/DAQ.hpp"
#include "Motor/NI.hpp"
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

#ifdef ARPAM_HAS_ALAZAR
private:
  // Buffer
  std::shared_ptr<RFBuffer<ArpamFloat>> m_buffer;

  // DAQ
  daq::DAQ *m_daq;
  QThread m_daqThread;

  // Motor
  motor::MotorNI *m_motor;
  QThread m_motorThread;

  // UI
  QPushButton *m_btnInitBoard;
  QPushButton *m_btnStartStopAcquisition;
};

#else
};

#endif // ARPAM_HAS_ALAZAR