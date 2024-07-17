#pragma once

#include "SaftParamsController.hpp"
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>
#include <uspam/beamformer/SAFT.hpp>

#include <QWidget>

class SaftParamsController : public QWidget {
  Q_OBJECT
public:
  using T = float;

  explicit SaftParamsController(QWidget *parent = nullptr);

signals:
  void paramsUpdated(uspam::beamformer::SaftDelayParams<T>);

private:
  inline void _paramsUpdatedInternal() { emit paramsUpdated(params); }

  uspam::beamformer::SaftDelayParams<T> params;
};