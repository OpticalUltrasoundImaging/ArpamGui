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
  uspam::recon::ReconParams2 params; // NOLINT(*non-private*)
  uspam::io::IOParams ioparams;      // NOLINT(*non-private*)

  explicit ReconParamsController(QWidget *parent = nullptr);

public slots:
  void resetParams2024v1() {
    params = uspam::recon::ReconParams2::system2024v1();
    ioparams = uspam::io::IOParams::system2024v1();
    updateGuiFromParams();
  }

  void resetParams2024v2probe1() {
    params = uspam::recon::ReconParams2::system2024v2probe1();
    ioparams = uspam::io::IOParams::system2024v2();
    updateGuiFromParams();
  }

  void resetParams2024v2probe2() {
    params = uspam::recon::ReconParams2::system2024v2probe2();
    ioparams = uspam::io::IOParams::system2024v2();
    updateGuiFromParams();
  }

  void resetParams2024v3() {
    params = uspam::recon::ReconParams2::system2024v3();
    ioparams = uspam::io::IOParams::system2024v3();
    updateGuiFromParams();
  }

signals:
  void paramsUpdated(uspam::recon::ReconParams2 params,
                     uspam::io::IOParams ioparams);

  void error(QString err);

private:
  std::vector<std::function<void()>> updateGuiFromParamsCallbacks;

  void updateGuiFromParams() {
    for (const auto &func : updateGuiFromParamsCallbacks) {
      func();
    }
  }
  inline void _paramsUpdatedInternal() { emit paramsUpdated(params, ioparams); }
};
