#pragma once

#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>
#include <QtDebug>
#include <QtLogging>
#include <uspam/io.hpp>
#include <uspam/recon.hpp>

class ReconParamsController : public QWidget {
  Q_OBJECT
public:
  uspam::recon::ReconParams2 params;
  uspam::io::IOParams ioparams;

  ReconParamsController(QWidget *parent = nullptr);

signals:
  void paramsUpdated(uspam::recon::ReconParams2 params,
                     uspam::io::IOParams ioparams);
  void error(QString err);

private:
  inline void _paramsUpdatedInternal() { emit paramsUpdated(params, ioparams); }

public slots:

  inline void noiseFloorPA_changed(int val) {
    params.noiseFloorPA = val;
    _paramsUpdatedInternal();
  }
  inline void noiseFloorUS_changed(int val) {
    params.noiseFloorUS = val;
    _paramsUpdatedInternal();
  }

  inline void dynamicRangePA_changed(int val) {
    params.desiredDynamicRangePA = val;
    _paramsUpdatedInternal();
  }
  inline void dynamicRangeUS_changed(int val) {
    params.desiredDynamicRangeUS = val;
    _paramsUpdatedInternal();
  }

  inline void rotOffset_changed(int val) {
    params.alineRotationOffset = val;
    _paramsUpdatedInternal();
  }

  inline void PAUSspacer_changed(int val) {
    ioparams.rf_size_spacer = val;
    _paramsUpdatedInternal();
  }
  inline void offsetUS_changed(int val) {
    ioparams.offsetUS = val;
    _paramsUpdatedInternal();
  }
  inline void offsetPA_changed(int val) {
    ioparams.offsetPA = val;
    _paramsUpdatedInternal();
  }
};