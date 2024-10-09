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
      uspam::beamformer::BeamformerParams<T> params =
          uspam::beamformer::BeamformerParams<T>::make_PA());

signals:
  void paramsUpdated(uspam::beamformer::BeamformerParams<T>);

private:
  inline void _paramsUpdatedInternal() { emit paramsUpdated(m_params); }

  uspam::beamformer::BeamformerParams<T> m_params;
};