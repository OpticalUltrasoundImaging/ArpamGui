#include "AcquisitionController.hpp"
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

AcquisitionController::AcquisitionController() : m_daq(new daq::DAQ) {
  // UI
  auto *layout = new QVBoxLayout;
  this->setLayout(layout);

  {
    auto *btn = new QPushButton("Init Alazar board");
    connect(btn, &QPushButton::pressed, this, [this]() {
      auto success = m_daq->initHardware();
      if (success) {
        QMessageBox::information(this, "Info", "Init hardware successful.");
      }
    });

    layout->addWidget(btn);
  }
};