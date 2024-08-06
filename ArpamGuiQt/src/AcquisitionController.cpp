#include "AcquisitionController.hpp"

#ifdef ARPAM_HAS_ALAZAR

#include "Motor/NI.hpp"
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <Qt>
#include <uspam/defer.h>

AcquisitionControllerObj::AcquisitionControllerObj(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer)
    : m_daq(buffer) {}

void AcquisitionControllerObj::startAcquisitionLoop() {
  acquiring = true;

  bool daqSuccess = true;
  bool motorSuccess = true;

  // Init DAQ board
  // Call the method directly to make sure sequential
  daqSuccess = m_daq.initHardware();

  if (daqSuccess) {
    daqSuccess = m_daq.prepareAcquisition();
  }

  if (daqSuccess) {
    emit acquisitionStarted();

    const auto maxFramePairs = 100;

    for (int pairsCompleted = 0; pairsCompleted < maxFramePairs && acquiring;
         ++pairsCompleted) {
      const int maxIdx = (pairsCompleted + 1) * 2;
      emit maxIndexChanged(maxIdx);

      // Start data acquisition for 2 BScans

      const auto motorMoveAsyncCB = [&]() {
        motorSuccess = m_motor.startMoveAsync();
      };

      // Scan clockwise
      {
        motorSuccess =
            m_motor.setDirection(motor::MotorNI::Direction::CLOCKWISE) &&
            m_motor.prepareMove();

        // First motor move failed to prepare, safe to break now.
        if (!motorSuccess) {
          break;
        }

        daqSuccess =
            m_daq.startAcquisition(1, pairsCompleted * 2, motorMoveAsyncCB);
        motorSuccess = m_motor.waitUntilMoveEnds();
      }

      // Scan anticlockwise
      {
        motorSuccess =
            motorSuccess &&
            m_motor.setDirection(motor::MotorNI::Direction::ANTICLOCKWISE) &&
            m_motor.prepareMove();

        if (daqSuccess) {
          daqSuccess = m_daq.startAcquisition(1, pairsCompleted * 2 + 1,
                                              motorMoveAsyncCB);
        } else {
          // Make sure motor turns back even if DAQ failed.
          motorMoveAsyncCB();
        }

        motorSuccess = m_motor.waitUntilMoveEnds();
      }

      if (!daqSuccess || !motorSuccess) {
        break;
      }
    }
  }

  if (!daqSuccess) {
    const auto &daqErr = m_daq.errMsg();
    if (!daqErr.isEmpty()) {
      emit error(daqErr);
      qCritical() << "Acquisition failed: " << daqErr;
    }
  }

  if (!motorSuccess) {
    const auto &motorErr = QString::fromLocal8Bit(m_motor.errMsg());
    if (!motorErr.isEmpty()) {
      emit error(motorErr);
      qCritical() << "Acquisition failed: " << motorErr;
    }
  }

  m_motor.abortMove(); // Clean up motor handles

  acquiring = false;
  m_daq.finishAcquisition();
  emit acquisitionFinished();
}

AcquisitionController::AcquisitionController(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer)
    : controller(buffer),

      m_btnStartStopAcquisition(new QPushButton("Start")),
      m_btnSaveDisplay(new QPushButton("Saving"))

{
  controller.moveToThread(&controllerThread);
  controllerThread.start(QThread::HighPriority);

  auto *layout = new QVBoxLayout;
  setLayout(layout);

  // Acquisition start/stop button
  // Save/display button
  {
    layout->addWidget(m_btnStartStopAcquisition);
    m_btnStartStopAcquisition->setStyleSheet("background-color: green");

    connect(m_btnStartStopAcquisition, &QPushButton::clicked, this, [this]() {
      m_btnStartStopAcquisition->setEnabled(false);

      m_btnSaveDisplay->setEnabled(false);
      m_motorTestGB->setEnabled(false);

      if (controller.isAcquiring()) {
        m_btnStartStopAcquisition->setText("Stopping");
        m_btnStartStopAcquisition->setStyleSheet("background-color: yellow");
        controller.stopAcquisitionLoop();
      } else {
        m_btnStartStopAcquisition->setText("Starting");
        QMetaObject::invokeMethod(
            &controller, &AcquisitionControllerObj::startAcquisitionLoop);
      }
    });

    const auto btnStateRunning = [this] {
      m_btnStartStopAcquisition->setEnabled(true);
      m_btnStartStopAcquisition->setText("Stop");
      m_btnStartStopAcquisition->setStyleSheet("background-color: red");
    };

    const auto btnStateStopped = [this] {
      m_btnStartStopAcquisition->setEnabled(true);
      m_btnStartStopAcquisition->setText("Start");
      m_btnStartStopAcquisition->setStyleSheet("background-color: green");

      m_btnSaveDisplay->setEnabled(true);
      m_motorTestGB->setEnabled(true);
    };

    connect(&controller, &AcquisitionControllerObj::acquisitionStarted, this,
            [btnStateRunning]() { btnStateRunning(); });

    connect(&controller, &AcquisitionControllerObj::acquisitionFinished, this,
            [btnStateStopped]() { btnStateStopped(); });

    connect(&controller, &AcquisitionControllerObj::error, this,
            [this](const QString &msg) {
              QMessageBox::information(this, "Acquisition controller", msg);
            });

    {
      layout->addWidget(m_btnSaveDisplay);

      m_btnSaveDisplay->setCheckable(true);
      m_btnSaveDisplay->setChecked(true);

      connect(m_btnSaveDisplay, &QPushButton::clicked, this,
              [this](bool checked) {
                if (checked) {
                  m_btnSaveDisplay->setText("Saving");
                  controller.daq().setSaveData(true);
                } else {
                  m_btnSaveDisplay->setText("Display only");
                  controller.daq().setSaveData(false);
                }
              });
    }
  }

  // Motor test buttons
  {
    m_motorTestGB = new QGroupBox("Motor testing");
    layout->addWidget(m_motorTestGB);

    auto *hlayout = new QHBoxLayout;
    m_motorTestGB->setLayout(hlayout);
    {
      auto *btn = new QPushButton("Clockwise");
      hlayout->addWidget(btn);

      connect(btn, &QPushButton::pressed, this,
              [this] { controller.motor().moveClockwise(); });
    }
    {
      auto *btn = new QPushButton("Anticlockwise");
      hlayout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this,
              [this] { controller.motor().moveAnticlockwise(); });
    }
  }
};

AcquisitionController::~AcquisitionController() {
  if (controllerThread.isRunning()) {
    controller.stopAcquisitionLoop();
    controllerThread.quit();
    controllerThread.wait();
  }
};

#else

AcquisitionController::AcquisitionController(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer) {}
AcquisitionController::~AcquisitionController() {}

#endif // ARPAM_HAS_ALAZAR
