#pragma once

#include "Common.hpp"
#include "DAQ/DAQ.hpp"
#include "Motor/MotorNI.hpp"
#include <QAction>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QPushButton>
#include <QSpinBox>
#include <QThread>
#include <QWidget>
#include <RFBuffer.hpp>
#include <atomic>
#include <memory>

#ifdef ARPAM_HAS_ALAZAR

struct AcquisitionParams {
  int maxFrames{400};        // Max number of frames to acquire.
  int scansEachDirection{1}; // scans in each direction
  double speed{1.0};         // rotations/sec
};

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
  void startAcquisitionLoop(AcquisitionParams params);
  void stopAcquisitionLoop() { acquiring = false; };

signals:
  void acquisitionStarted();
  void acquisitionFinished();

  void error(QString);

private:
  motor::MotorNI m_motor;
  daq::DAQ m_daq;

  std::atomic<bool> acquiring{false};
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

  auto actShowMotorTestPanel() const { return m_actShowMotorTestPanel; }

  void startAcquisitionBackAndForth();

  // Acquisition controller, thread, and parameters
  AcquisitionControllerObj controller;
  QThread controllerThread;
  AcquisitionParams m_acqParams;

  // Actions
  QAction *m_actShowMotorTestPanel;

  // UI
  QPushButton *m_btnStartStopAcquisition;
  QPushButton *m_btnSaveDisplay;

  // Acquisition params
  QSpinBox *m_sbMaxFrames;
  QSpinBox *m_sbScansEachDirection;
  QDoubleSpinBox *m_sbSpeed;

  QGroupBox *m_motorTestGB;
};

#else
};

#endif // ARPAM_HAS_ALAZAR