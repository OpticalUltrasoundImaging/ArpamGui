#pragma once

#include "Common.hpp"
#include "DAQ/DAQ.hpp"
#include "Motor/NI.hpp"
#include <QGroupBox>
#include <QPushButton>
#include <QThread>
#include <QWidget>
#include <RFBuffer.hpp>
#include <atomic>
#include <memory>
#include <qgroupbox.h>

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
  ~AcquisitionControllerObj() {}

  [[nodiscard]] auto &daq() { return m_daq; }
  [[nodiscard]] auto &motor() { return m_motor; }

  bool isAcquiring() const { return acquiring; }
  void startAcquisitionLoop();
  void stopAcquisitionLoop() { shouldStop = true; };

signals:
  void acquisitionStarted();
  void acquisitionFinished();
  void acquisitionFailed();

  void maxIndexChanged(int);

  void error(QString);

private:
  motor::MotorNI m_motor;
  daq::DAQ m_daq;

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
  QPushButton *m_btnStartStopAcquisition;
  QPushButton *m_btnSaveDisplay;

  QGroupBox *m_motorTestGB;
};

#else
};

#endif // ARPAM_HAS_ALAZAR