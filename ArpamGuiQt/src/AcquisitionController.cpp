#include "AcquisitionController.hpp"
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <qnamespace.h>
#include <qobjectdefs.h>

AcquisitionController::AcquisitionController() : m_daq(new daq::DAQ) {

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

  {
    auto *btn = new QPushButton("Init Alazar board");
    layout->addWidget(btn);
    connect(btn, &QPushButton::pressed, m_daq, &daq::DAQ::initHardware);
  }

  {
    auto *btn = new QPushButton("Start");
    layout->addWidget(btn);
    connect(btn, &QPushButton::pressed, this, [this, btn]() {
      btn->setDisabled(true);
      if (m_daq->isAcquiring()) {
        m_daq->stopAcquisition();
      } else {
        QMetaObject::invokeMethod(m_daq, &daq::DAQ::startAcquisition);
      }
    });

    connect(m_daq, &daq::DAQ::acquisitionStarted, this, [this, btn]() {
      btn->setEnabled(true);
      btn->setText("Stop");
    });

    connect(m_daq, &daq::DAQ::acquisitionStopped, this, [this, btn]() {
      btn->setEnabled(true);
      btn->setText("Start");
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