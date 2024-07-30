#pragma once

#include "DAQ/DAQ.hpp"
#include <QThread>
#include <QWidget>
#include <qevent.h>

/*
This widget acts as the data acquisition control UI

*/
class AcquisitionController : public QWidget {
  Q_OBJECT
public:
  AcquisitionController();
  AcquisitionController(const AcquisitionController &) = delete;
  AcquisitionController(AcquisitionController &&) = delete;
  AcquisitionController &operator=(const AcquisitionController &) = delete;
  AcquisitionController &operator=(AcquisitionController &&) = delete;
  ~AcquisitionController() override;

private:
  daq::DAQ *m_daq;
  QThread m_daqThread;
};