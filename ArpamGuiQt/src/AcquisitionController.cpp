#include "AcquisitionController.hpp"
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <Qt>

AcquisitionController::AcquisitionController(
    const std::shared_ptr<RFBuffer<ArpamFloat>> &buffer)
    : m_buffer(buffer), m_daq(new daq::DAQ(buffer)),

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
        QMetaObject::invokeMethod(m_daq, &daq::DAQ::startAcquisition);
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
  }
};

AcquisitionController::~AcquisitionController() {
  if (m_daqThread.isRunning()) {
    m_daq->stopAcquisition();
    m_daqThread.quit();
    m_daqThread.wait();
  }
};