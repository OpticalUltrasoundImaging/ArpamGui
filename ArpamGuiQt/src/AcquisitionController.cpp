#include "AcquisitionController.hpp"
#include <qnamespace.h>
#include <qobjectdefs.h>

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
    : m_daq(new daq::DAQ(buffer)), m_motor(new motor::MotorNI) {

  {
    m_daq->moveToThread(&m_daqThread);
    connect(&m_daqThread, &QThread::finished, m_daq, &QObject::deleteLater);

    m_daqThread.start();
  }
  m_motor->setParent(this);

  dumpObjectTree();
}

void AcquisitionControllerObj::startAcquisition() {
  shouldStop = false;
  acquiring = true;
  defer { acquiring = false; };

  // Init DAQ board
  // Call the method directly to make sure sequential
  m_daq->initHardware();
  // QMetaObject::invokeMethod(m_daq, &daq::DAQ::initHardware,
  //                           Qt::BlockingQueuedConnection);

  const auto maxFramePairs = 100;
  for (int i = 0; i < maxFramePairs && !shouldStop; ++i) {
    const int maxIdx = (i + 1) * 2;
    emit maxIndexChanged(maxIdx);

    // Start data acquisition for 2 BScans in DAQ thread
    QMetaObject::invokeMethod(m_daq, &daq::DAQ::startAcquisition, 2, i * 2);

    // Start motor rotation for 1 back-and-forth rotation in the motor thread
    m_motor->moveClockwiseThenAnticlockwise();
  }

  if (shouldStop) {
    QMetaObject::invokeMethod(m_daq, &daq::DAQ::stopAcquisition);
  }
}

AcquisitionController::AcquisitionController(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer)
    : controller(buffer),
      m_btnInitBoard(new QPushButton("Initialize Alazar Board")),
      m_btnStartStopAcquisition(new QPushButton("Start"))

{
  controller.moveToThread(&controllerThread);
  controllerThread.start(QThread::HighPriority);

  connect(
      controller.daq(), &daq::DAQ::messageBox, this,
      [this](const QString &msg) {
        QMessageBox::information(this, "DAQ Info", msg);
      },
      Qt::BlockingQueuedConnection);

  connect(
      controller.motor(), &motor::MotorNI::messageBox, this,
      [this](const QString &msg) {
        QMessageBox::information(this, "MotorNI Info", msg);
      },
      Qt::BlockingQueuedConnection);
  /*
  UI
  */
  auto *layout = new QVBoxLayout;
  this->setLayout(layout);

  // Acquisition start/stop button
  {
    layout->addWidget(m_btnStartStopAcquisition);
    m_btnStartStopAcquisition->setStyleSheet("background-color: green");

    connect(m_btnStartStopAcquisition, &QPushButton::clicked, this, [this]() {
      m_btnStartStopAcquisition->setEnabled(false);
      if (controller.isAcquiring()) {
        controller.stopAcquisition();
      } else {
        QMetaObject::invokeMethod(&controller,
                                  &AcquisitionControllerObj::startAcquisition);
      }
    });

    connect(controller.daq(), &daq::DAQ::acquisitionStarted, this, [this]() {
      // m_btnStartStopAcquisition->setEnabled(true);
      m_btnStartStopAcquisition->setText("Stop");
      m_btnStartStopAcquisition->setStyleSheet("background-color: red");
    });

    connect(controller.daq(), &daq::DAQ::acquisitionStopped, this, [this]() {
      m_btnStartStopAcquisition->setEnabled(true);
      m_btnStartStopAcquisition->setText("Start");
      m_btnStartStopAcquisition->setStyleSheet("background-color: green");
    });
  }

  // Motor test buttons
  {
    auto *gb = new QGroupBox("Motor testing");
    layout->addWidget(gb);

    auto *hlayout = new QHBoxLayout;
    gb->setLayout(hlayout);
    {
      auto *btn = new QPushButton("Clockwise");
      hlayout->addWidget(btn);
      connect(btn, &QPushButton::pressed, controller.motor(),
              &motor::MotorNI::moveClockwise);
    }
    {
      auto *btn = new QPushButton("Anticlockwise");
      hlayout->addWidget(btn);
      connect(btn, &QPushButton::pressed, controller.motor(),
              &motor::MotorNI::moveAnticlockwise);
    }
    {
      auto *btn = new QPushButton("Clockwise then anti");
      hlayout->addWidget(btn);
      connect(btn, &QPushButton::pressed, controller.motor(),
              &motor::MotorNI::moveClockwiseThenAnticlockwise);
    }
  }
};

AcquisitionController::~AcquisitionController() {
  if (controllerThread.isRunning()) {
    controller.stopAcquisition();
    controllerThread.quit();
    controllerThread.wait();
  }
};

#else

AcquisitionController::AcquisitionController(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer) {}
AcquisitionController::~AcquisitionController() {}

#endif // ARPAM_HAS_ALAZAR
