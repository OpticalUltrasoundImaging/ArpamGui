#include "AcquisitionController.hpp"
#include <qgridlayout.h>
#include <qmessagebox.h>
#include <qpushbutton.h>

#ifdef ARPAM_HAS_ALAZAR

#include "Motor/NI.hpp"
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
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

    const auto maxFramePairs = (m_maxFrames + 1) / 2;

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

      m_actShowMotorTestPanel(new QAction("Motor Test Panel")),

      m_btnStartStopAcquisition(new QPushButton("Start")),
      m_btnSaveDisplay(new QPushButton("Saving")), m_spMaxFrames(new QSpinBox),

      m_motorTestGB(new QGroupBox("Motor testing"))

{
  controller.moveToThread(&controllerThread);
  controllerThread.start(QThread::HighPriority);

  auto *layout = new QVBoxLayout;
  setLayout(layout);

  auto *acqGrid = new QGridLayout;
  layout->addLayout(acqGrid);

  // Acquisition start/stop button
  // Save/display button
  {
    acqGrid->addWidget(m_btnStartStopAcquisition, 0, 0);
    m_btnStartStopAcquisition->setStyleSheet("background-color: green");

    connect(m_btnStartStopAcquisition, &QPushButton::clicked, this, [this]() {
      m_btnStartStopAcquisition->setEnabled(false);
      m_btnSaveDisplay->setEnabled(false);
      m_spMaxFrames->setEnabled(false);

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
      m_spMaxFrames->setEnabled(true);
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
  }

  // Button to toggle between saving/display only
  {
    acqGrid->addWidget(m_btnSaveDisplay, 0, 1);

    m_btnSaveDisplay->setCheckable(true);

    connect(m_btnSaveDisplay, &QPushButton::toggled, this,
            [this](bool checked) {
              if (checked) {
                m_btnSaveDisplay->setText("Saving");
                m_btnSaveDisplay->setStyleSheet("background-color: green");

                controller.daq().setSaveData(true);
              } else {
                m_btnSaveDisplay->setText("Display only");
                m_btnSaveDisplay->setStyleSheet("");

                controller.daq().setSaveData(false);
              }
            });
    m_btnSaveDisplay->setChecked(true);
  }

  // Spinbox to set maxFrames
  {
    auto *lbl = new QLabel("Max frames");
    acqGrid->addWidget(lbl, 1, 0);

    acqGrid->addWidget(m_spMaxFrames, 1, 1);

    m_spMaxFrames->setMinimum(2);
    m_spMaxFrames->setMaximum(1000);
    m_spMaxFrames->setSingleStep(1);

    m_spMaxFrames->setValue(controller.maxFrames());

    connect(m_spMaxFrames, &QSpinBox::valueChanged, &controller,
            &AcquisitionControllerObj::setMaxFrames);
  }

  // Motor test panel
  {
    m_actShowMotorTestPanel->setCheckable(true);
    m_actShowMotorTestPanel->setChecked(false);
    connect(m_actShowMotorTestPanel, &QAction::triggered, this,
            [this](bool checked) { m_motorTestGB->setVisible(checked); });
    m_motorTestGB->setVisible(false);
  }

  {
    layout->addWidget(m_motorTestGB);
    auto *hlayout = new QHBoxLayout;
    m_motorTestGB->setLayout(hlayout);
    {
      auto *btn = new QPushButton("Clockwise");
      hlayout->addWidget(btn);

      connect(btn, &QPushButton::pressed, this, [this] {
        const auto success = controller.motor().moveClockwise();
        if (!success) {
          QMessageBox::information(
              this, "Motor",
              QString::fromLocal8Bit(controller.motor().errMsg()));
        }
      });
    }
    {
      auto *btn = new QPushButton("Anticlockwise");
      hlayout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this, [this] {
        const auto success = controller.motor().moveAnticlockwise();
        if (!success) {
          QMessageBox::information(
              this, "Motor",
              QString::fromLocal8Bit(controller.motor().errMsg()));
        }
      });
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
