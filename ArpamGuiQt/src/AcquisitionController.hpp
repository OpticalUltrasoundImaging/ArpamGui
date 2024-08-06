#pragma once

#include "Common.hpp"
#include "DAQ/DAQ.hpp"
#include "Motor/NI.hpp"
#include <QPushButton>
#include <QThread>
#include <QWidget>
#include <RFBuffer.hpp>
#include <atomic>
#include <memory>

#ifdef ARPAM_HAS_ALAZAR

class AcquisitionControllerObj : public QObject {
  Q_OBJECT
public:
  explicit AcquisitionControllerObj(
      const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer);
  AcquisitionControllerObj(const AcquisitionControllerObj &) = delete;
  AcquisitionControllerObj(AcquisitionControllerObj &&) = delete;
  AcquisitionControllerObj &
  operator=(const AcquisitionControllerObj &) = delete;
  AcquisitionControllerObj &operator=(AcquisitionControllerObj &&) = delete;
  ~AcquisitionControllerObj() {
    if (m_daqThread.isRunning()) {
      m_daq->stopAcquisition();
      m_daqThread.quit();
      m_daqThread.wait();
    }

    // if (m_motorThread.isRunning()) {
    //   m_motorThread.quit();
    //   m_motorThread.wait();
    // }
  }

  [[nodiscard]] auto daq() const { return m_daq; }
  [[nodiscard]] auto motor() const { return m_motor; }

  bool isAcquiring() const { return acquiring; }
  void startAcquisition();
  void stopAcquisition() { shouldStop = true; };

signals:
  void maxIndexChanged(int);

private:
  // DAQ
  daq::DAQ *m_daq;
  QThread m_daqThread;

  // Motor
  motor::MotorNI *m_motor;
  // QThread m_motorThread;

  std::atomic<bool> acquiring{false};
  std::atomic<bool> shouldStop{false};
};

#endif // ARPAM_HAS_ALAZAR

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

  void startAcquisitionBackAndForth();

  AcquisitionControllerObj controller;
  QThread controllerThread;

  // UI
  QPushButton *m_btnInitBoard;
  QPushButton *m_btnStartStopAcquisition;
};

#else
};

#endif // ARPAM_HAS_ALAZAR