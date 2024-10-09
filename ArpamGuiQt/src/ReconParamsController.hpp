#pragma once

#include <Common.hpp>
#include <QWidget>
#include <QtDebug>
#include <uspam/io.hpp>
#include <uspam/recon.hpp>

class ReconParamsController : public QWidget {
  Q_OBJECT
public:
  uspam::recon::ReconParams2 params; // NOLINT(*non-private*)
  uspam::io::IOParams ioparams;      // NOLINT(*non-private*)

  explicit ReconParamsController(QWidget *parent = nullptr);

public slots:
  void resetParams2024v1();
  void resetParams2024v2GUI();

signals:
  void paramsUpdated(uspam::recon::ReconParams2 params,
                     uspam::io::IOParams ioparams);

  void error(QString err);

private:
  void updateGuiFromParams();
  inline void _paramsUpdatedInternal() { emit paramsUpdated(params, ioparams); }

  std::vector<std::function<void()>> updateGuiFromParamsCallbacks;
};
