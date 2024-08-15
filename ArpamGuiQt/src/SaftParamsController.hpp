#pragma once

#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QLabel>
#include <uspam/beamformer/SAFT.hpp>

#include <QWidget>

class SaftParamsController : public QWidget {
  Q_OBJECT
public:
  using T = float;

  explicit SaftParamsController(
      uspam::beamformer::SaftDelayParams<T> params =
          uspam::beamformer::SaftDelayParams<T>::make_PA());

signals:
  void paramsUpdated(uspam::beamformer::SaftDelayParams<T>);

private:
  inline void _paramsUpdatedInternal() { emit paramsUpdated(m_params); }

  uspam::beamformer::SaftDelayParams<T> m_params;
};