#include "AcquisitionController.hpp"

#ifdef ARPAM_HAS_ALAZAR
#include "Motor/NI.hpp"
#include <QDoubleSpinBox>
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

void AcquisitionControllerObj::startAcquisitionLoop(AcquisitionParams params) {
  acquiring = true;

  bool daqSuccess = true;
  bool motorSuccess = true;

  // Init DAQ board
  // Call the method directly to make sure sequential
  daqSuccess = m_daq.initHardware();
  if (daqSuccess) {
    daqSuccess = m_daq.prepareAcquisition();
  }

  /*
  The motor must return back to zero, so after n clockwise rotation(s), it must
  do n anticlockwise rotation(s). We call this full +n -n a sequence
  */
  const int scansPerSequence = params.scansEachDirection * 2;
  params.maxFrames = std::max(params.maxFrames, scansPerSequence);

  // Init the motor move signal
  using Direction = motor::MotorNI::Direction;
  motorSuccess = m_motor.prepareMove(params.speed, params.scansEachDirection);

  const auto motorMoveAsyncCB = [this, &motorSuccess]() {
    motorSuccess = m_motor.startMoveAsync();
  };

  if (daqSuccess && motorSuccess) {
    emit acquisitionStarted();

    const auto maxSequences = params.maxFrames / scansPerSequence;

    for (int sequencesComplete = 0;
         sequencesComplete < maxSequences && acquiring; ++sequencesComplete) {
      auto currIdx = sequencesComplete * scansPerSequence;

      //// Start data acquisition scansEachDirection
      // Scan clockwise
      {
        motorSuccess = m_motor.setDirection(Direction::CLOCKWISE);
        // First motor move failed to prepare, safe to break now.
        if (!motorSuccess) {
          break;
        }

        daqSuccess =
            m_daq.acquire(params.scansEachDirection, currIdx, motorMoveAsyncCB);
        motorSuccess = m_motor.waitUntilMoveEnds();

        if (!motorSuccess) {
          break;
        }
      }

      // Scan anticlockwise
      currIdx += params.scansEachDirection;
      {
        motorSuccess = m_motor.setDirection(Direction::ANTICLOCKWISE);

        if (daqSuccess) {
          daqSuccess = m_daq.acquire(params.scansEachDirection, currIdx,
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
    const auto &daqErr = "DAQ error: " + m_daq.errMsg();
    if (!daqErr.isEmpty()) {
      emit error(daqErr);
      qCritical() << "Acquisition failed: " << daqErr;
    }
  }

  if (!motorSuccess) {
    const auto &motorErr =
        "Motor error: " + QString::fromLocal8Bit(m_motor.errMsg());
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

      m_spScansEachDirection(new QSpinBox), m_spSpeed(new QDoubleSpinBox),

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
            &controller, &AcquisitionControllerObj::startAcquisitionLoop,
            m_acqParams);
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
    m_spMaxFrames->setSingleStep(10);
    m_spMaxFrames->setValue(m_acqParams.maxFrames);

    connect(m_spMaxFrames, &QSpinBox::valueChanged,
            [this](int val) { m_acqParams.maxFrames = val; });
  }

  // Acquisition parameters
  {
    // scansEachDirection
    {
      auto *lbl = new QLabel("Scans each direction");
      acqGrid->addWidget(lbl, 2, 0);
      acqGrid->addWidget(m_spScansEachDirection, 2, 1);

      m_spScansEachDirection->setMinimum(1);
      m_spScansEachDirection->setMaximum(10);
      m_spScansEachDirection->setSingleStep(1);

      m_spScansEachDirection->setValue(m_acqParams.scansEachDirection);

      connect(m_spScansEachDirection, &QSpinBox::valueChanged,
              [this](int val) { m_acqParams.scansEachDirection = val; });
    }

    // Speed
    {
      // auto *lbl=new QLabel("Speed");
    }
  }

  /**
    Motor test panel
   */
  {
    m_actShowMotorTestPanel->setCheckable(true);
    m_actShowMotorTestPanel->setChecked(false);
    connect(m_actShowMotorTestPanel, &QAction::triggered, this,
            [this](bool checked) { m_motorTestGB->setVisible(checked); });
    m_motorTestGB->setVisible(false);
  }

  {
    layout->addWidget(m_motorTestGB);
    auto *gridLayout = new QGridLayout;
    m_motorTestGB->setLayout(gridLayout);

    int row = 0;

    auto *rotationsSpinBox = new QDoubleSpinBox;
    {
      gridLayout->addWidget(rotationsSpinBox, row, 1);
      rotationsSpinBox->setMaximum(20.0);
      rotationsSpinBox->setMinimum(0.0);
      rotationsSpinBox->setValue(1.0);
      rotationsSpinBox->setSingleStep(1.0);
      auto *label = new QLabel("Rotations");
      gridLayout->addWidget(label, row++, 0);
    }

    auto *speedSpinBox = new QDoubleSpinBox;
    {
      gridLayout->addWidget(speedSpinBox, row, 1);
      speedSpinBox->setMaximum(4);
      speedSpinBox->setMinimum(0);
      speedSpinBox->setValue(1);
      speedSpinBox->setSingleStep(0.2);
      auto *label = new QLabel("Speed");
      gridLayout->addWidget(label, row++, 0);
    }

    {
      auto *btn = new QPushButton("Clockwise");
      gridLayout->addWidget(btn, row, 0);

      connect(btn, &QPushButton::pressed, this,
              [this, speedSpinBox, rotationsSpinBox] {
                const auto success = controller.motor().moveClockwise(
                    speedSpinBox->value(), rotationsSpinBox->value());

                if (!success) {
                  QMessageBox::information(
                      this, "Motor",
                      QString::fromLocal8Bit(controller.motor().errMsg()));
                }
              });
    }

    {
      auto *btn = new QPushButton("Anticlockwise");
      gridLayout->addWidget(btn, row, 1);
      connect(btn, &QPushButton::pressed, this,
              [this, speedSpinBox, rotationsSpinBox] {
                const auto success = controller.motor().moveAnticlockwise(
                    speedSpinBox->value(), rotationsSpinBox->value());

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
