#include "AcquisitionController.hpp"
#include "Motor/NI.hpp"

#ifdef ARPAM_HAS_ALAZAR

#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <Qt>

AcquisitionController::AcquisitionController(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer)
    : m_buffer(buffer),

      m_daq(new daq::DAQ(buffer)),

      m_motor(new motor::MotorNI),

      m_btnInitBoard(new QPushButton("Initialize Alazar Board")),
      m_btnStartStopAcquisition(new QPushButton("Start"))

{

  {
    m_daq->moveToThread(&m_daqThread);

    connect(&m_daqThread, &QThread::finished, m_daq, &QObject::deleteLater);

    connect(
        m_daq, &daq::DAQ::messageBox, this,
        [this](QString msg) {
          QMessageBox::information(this, "DAQ Info", msg);
        },
        Qt::BlockingQueuedConnection);

    m_daqThread.start();
  }

  {
    m_motor->moveToThread(&m_motorThread);

    connect(&m_motorThread, &QThread::finished, m_motor, &QObject::deleteLater);

    connect(
        m_motor, &motor::MotorNI::messageBox, this,
        [this](QString msg) {
          QMessageBox::information(this, "MotorNI Info", msg);
        },
        Qt::BlockingQueuedConnection);

    m_motorThread.start();
  }

  // UI
  auto *layout = new QVBoxLayout;
  this->setLayout(layout);

  // Initialize Alazar board button
  {
    layout->addWidget(m_btnInitBoard);
    connect(m_btnInitBoard, &QPushButton::pressed, m_daq,
            &daq::DAQ::initHardware);
  }

  // Acquisition start/stop button
  {
    layout->addWidget(m_btnStartStopAcquisition);
    connect(m_btnStartStopAcquisition, &QPushButton::pressed, this, [this]() {
      m_btnStartStopAcquisition->setDisabled(true);
      if (m_daq->isAcquiring()) {
        m_daq->stopAcquisition();
      } else {
        QMetaObject::invokeMethod(m_daq, &daq::DAQ::startAcquisition, 2);
        QMetaObject::invokeMethod(
            m_motor, &motor::MotorNI::moveClockwiseThenAnticlockwise);
      }
    });

    connect(m_daq, &daq::DAQ::acquisitionStarted, this, [this]() {
      m_btnStartStopAcquisition->setEnabled(true);
      m_btnStartStopAcquisition->setText("Stop");
    });

    connect(m_daq, &daq::DAQ::acquisitionStopped, this, [this]() {
      m_btnStartStopAcquisition->setEnabled(true);
      m_btnStartStopAcquisition->setText("Start");
    });

    m_btnStartStopAcquisition->setEnabled(false);
    connect(m_daq, &daq::DAQ::initHardwareSuccessful, this, [this] {
      m_btnStartStopAcquisition->setEnabled(true);
    });
  }

  // Motor test buttons
  {
    auto *btn = new QPushButton("Motor clockwise");
    layout->addWidget(btn);
    connect(btn, &QPushButton::pressed, m_motor,
            &motor::MotorNI::moveClockwise);
  }
  {
    auto *btn = new QPushButton("Motor anticlockwise");
    layout->addWidget(btn);
    connect(btn, &QPushButton::pressed, m_motor,
            &motor::MotorNI::moveAnticlockwise);
  }
  {
    auto *btn = new QPushButton("Motor clockwise then anti");
    layout->addWidget(btn);
    connect(btn, &QPushButton::pressed, m_motor,
            &motor::MotorNI::moveClockwiseThenAnticlockwise);
  }
};

AcquisitionController::~AcquisitionController() {
  if (m_daqThread.isRunning()) {
    m_daq->stopAcquisition();
    m_daqThread.quit();
    m_daqThread.wait();
  }

  if (m_motorThread.isRunning()) {
    m_motorThread.quit();
    m_motorThread.wait();
  }
};

#else

AcquisitionController::AcquisitionController(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer) {}
AcquisitionController::~AcquisitionController() {}

#endif // ARPAM_HAS_ALAZAR