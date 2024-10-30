#pragma once

#include "Common.hpp"
#include "DAQ/DAQ.hpp"
#include "Motor/NI.hpp"
#include <QAction>
#include <QGroupBox>
#include <QPushButton>
#include <QSpinBox>
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
  void stopAcquisitionLoop() { acquiring = false; };

  [[nodiscard]] auto maxFrames() const { return m_maxFrames; }

public slots:
  void setMaxFrames(int val) { m_maxFrames = val; }

signals:
  void acquisitionStarted();
  void acquisitionFinished();

  void maxIndexChanged(int);

  void error(QString);

private:
  motor::MotorNI m_motor;
  daq::DAQ m_daq;

  int m_maxFrames{400};

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

  AcquisitionControllerObj controller;
  QThread controllerThread;

  // Actions
  QAction *m_actShowMotorTestPanel;

  // UI
  QPushButton *m_btnStartStopAcquisition;
  QPushButton *m_btnSaveDisplay;

  QSpinBox *m_spMaxFrames;

  QGroupBox *m_motorTestGB;
};

#else
};

#endif // ARPAM_HAS_ALAZAR