#pragma once

#include "DAQ/DAQ.hpp"
#include <QWidget>

/*
This widget acts as the data acquisition control UI

*/
class AcquisitionController : public QWidget {
  Q_OBJECT
public:
  AcquisitionController();

private:
  daq::DAQ *m_daq;
};