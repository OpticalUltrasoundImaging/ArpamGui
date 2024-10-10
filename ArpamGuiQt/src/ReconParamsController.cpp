#include "ReconParamsController.hpp"
#include "CollapsibleGroupBox.hpp"
#include "Common.hpp"
#include "uspam/reconParams.hpp"
#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QIntValidator>
#include <QLabel>
#include <QPushButton>
#include <QRegularExpression>
#include <QSpinBox>
#include <QValidator>
#include <QVariant>
#include <Qt>
#include <vector>

// NOLINTBEGIN(*-magic-numbers)

ReconParamsController::ReconParamsController(QWidget *parent)
    : QWidget(parent),
      params(uspam::recon::ReconParams2::system2024v2GUIprobe2()),
      ioparams(uspam::io::IOParams::system2024v2GUI()) {
  auto *layout = new QVBoxLayout();
  this->setLayout(layout);

  const auto makeQSpinBox = [this](int &value, const std::pair<int, int> &range,
                                   const int step = 1) {
    auto *spinBox = new QSpinBox;
    spinBox->setValue(value);
    spinBox->setRange(range.first, range.second);
    spinBox->setSingleStep(step);
    connect(spinBox, &QSpinBox::valueChanged, this, [&](int newValue) {
      value = newValue;
      this->_paramsUpdatedInternal();
    });
    return spinBox;
  };

  const auto makeQSpinBox2 = [this](int &value, int &value2,
                                    const std::pair<int, int> &range,
                                    const int step = 1) {
    value2 = value;
    auto *spinBox = new QSpinBox;
    spinBox->setValue(value);
    spinBox->setRange(range.first, range.second);
    spinBox->setSingleStep(step);
    connect(spinBox, &QSpinBox::valueChanged, this,
            [this, &value, &value2](const int newValue) {
              value = newValue;
              this->_paramsUpdatedInternal();
            });
    return spinBox;
  };

  const auto makeQDoubleSpinBox =
      [this]<typename Float>(const std::pair<double, double> &range,
                             const double singleStep, Float &value,
                             const double scalar = 1.0) {
        auto *spinBox = new QDoubleSpinBox;
        spinBox->setValue(static_cast<double>(value) / scalar);
        spinBox->setRange(range.first, range.second);
        spinBox->setSingleStep(singleStep);
        connect(spinBox, &QDoubleSpinBox::valueChanged, this,
                [this, &value, scalar](const double newValue) {
                  value = static_cast<Float>(newValue * scalar);
                  this->_paramsUpdatedInternal();
                });
        return spinBox;
      };

  const auto makeQDoubleSpinBox2 =
      [this]<typename Float>(const std::pair<double, double> &range,
                             double singleStep, Float &value1, Float &value2,
                             double scalar = 1) {
        value2 = value1;
        auto *spinBox = new QDoubleSpinBox;
        spinBox->setRange(range.first, range.second);
        spinBox->setSingleStep(singleStep);
        spinBox->setValue(static_cast<double>(value1) / scalar);
        connect(spinBox, &QDoubleSpinBox::valueChanged, this,
                [this, &value1, &value2, scalar](const double newValue) {
                  value1 = static_cast<Float>(newValue * scalar);
                  value2 = static_cast<Float>(newValue * scalar);
                  this->_paramsUpdatedInternal();
                });
        return spinBox;
      };

  const auto makeLabeledSpinbox =
      [this, &makeQSpinBox](QGridLayout *layout, int row, const QString &name,
                            const QString &desc, const QString &suffix,
                            int &value, const std::pair<int, int> &range,
                            const int step = 1) {
        auto *label = new QLabel(name);
        label->setToolTip(desc);
        layout->addWidget(label, row, 0);

        auto *sp = makeQSpinBox(value, range, step);
        sp->setSuffix(suffix);
        layout->addWidget(sp, row, 1);

        updateGuiFromParamsCallbacks.emplace_back(
            [this, sp, &value] { sp->setValue(value); });

        return std::tuple{label, sp};
      };

  const auto makeLabeledSpinbox2 =
      [this, &makeQSpinBox2](
          QGridLayout *layout, int row, const QString &name,
          const QString &desc, const QString &suffix, int &value, int &value2,
          const std::pair<int, int> &range, const int step = 1) {
        auto *label = new QLabel(name);
        label->setToolTip(desc);
        layout->addWidget(label, row, 0);

        auto *sp = makeQSpinBox2(value, value2, range, step);
        sp->setSuffix(suffix);
        layout->addWidget(sp, row, 1);

        updateGuiFromParamsCallbacks.emplace_back(
            [this, sp, &value] { sp->setValue(value); });

        return std::tuple{label, sp};
      };

  const auto makeLabeledDoubleSpinbox =
      [this, &makeQDoubleSpinBox]<typename Float>(
          QGridLayout *layout, int row, const QString &name,
          const QString &desc, const QString &suffix, Float &value,
          const std::pair<double, double> &range, const double step,
          const double scalar = 1.0) {
        auto *label = new QLabel(name);
        label->setToolTip(desc);
        layout->addWidget(label, row, 0);

        auto *sp = makeQDoubleSpinBox(range, step, value, scalar);
        sp->setSuffix(suffix);
        layout->addWidget(sp, row, 1);

        updateGuiFromParamsCallbacks.emplace_back(
            [this, sp, &value, scalar] { sp->setValue(value / scalar); });

        return std::tuple{label, sp};
      };

  const auto makeLabeledDoubleSpinbox2 =
      [this, &makeQDoubleSpinBox2]<typename Float>(
          QGridLayout *layout, int row, const QString &name,
          const QString &desc, const QString &suffix, Float &value1,
          Float &value2, const std::pair<double, double> &range,
          const double step, const double scalar = 1.0) {
        auto *label = new QLabel(name);
        label->setToolTip(desc);
        layout->addWidget(label, row, 0);

        auto *sp = makeQDoubleSpinBox2(range, step, value1, value2, scalar);
        sp->setSuffix(suffix);
        layout->addWidget(sp, row, 1);

        updateGuiFromParamsCallbacks.emplace_back(
            [this, sp, &value1, scalar] { sp->setValue(value1 / scalar); });

        return std::tuple{label, sp};
      };

  const auto makeLabeledCheckbox = [this](QGridLayout *layout, int row,
                                          const QString &name,
                                          const QString &desc, bool &value) {
    layout->addWidget(new QLabel(name), row, 0);

    auto *cb = new QCheckBox();
    layout->addWidget(cb, row, 1);
    using Qt::CheckState;

    connect(cb, &QCheckBox::checkStateChanged, this,
            [this, &value](CheckState state) {
              value = state == CheckState::Checked;
              _paramsUpdatedInternal();
            });

    updateGuiFromParamsCallbacks.emplace_back([this, cb, &value] {
      cb->setCheckState(value ? Qt::CheckState::Checked
                              : Qt::CheckState::Unchecked);
    });
  };

  // Presets
  {
    auto *gb = new CollapsibleGroupBox("Presets");
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);

    {
      auto *btn = new QPushButton("Legacy Labview");
      layout->addWidget(btn, 0, 0);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v1);
    }

    {
      auto *btn = new QPushButton("Probe 1");
      layout->addWidget(btn, 1, 0);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v2GUI);
    }

    {
      auto *btn = new QPushButton("Probe 2");
      layout->addWidget(btn, 2, 0);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v2GUIprobe2);
    }
  }

  // System parameters
  {
    auto *gb = new CollapsibleGroupBox("System");
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 0;

    // Sound speed
    makeLabeledDoubleSpinbox(layout, row++, "Sound speed", "", " m/s",
                             params.system.soundSpeed, {1300.0, 1700.0}, 10.0);

    // Sampling freq
    makeLabeledDoubleSpinbox(
        layout, row++, "Sampling Freq",
        "This only affects reconstruction and doesn't change the acquisition",
        " MHz", params.system.fs, {0.0, 200.0}, 10.0, 1.0E6);

    // Imaging head
    makeLabeledDoubleSpinbox(layout, row++, "Transducer offset", "", " mm",
                             params.system.transducerOffset, {0.0, 20.0}, 0.1);

    makeLabeledDoubleSpinbox(layout, row++, "Focal length", "", " mm",
                             params.system.focalLength, {0.0, 25.0}, 0.1);

    makeLabeledDoubleSpinbox(layout, row++, "Transducer diameter", "", " mm",
                             params.system.transducerDiameter, {0.0, 25.0},
                             0.1);

    makeLabeledDoubleSpinbox(layout, row++, "Illumination angle", "", " deg",
                             params.system.illumAngleDeg, {0.0, 25.0}, 0.1);

    makeLabeledCheckbox(layout, row++, "Flip on even",
                        "Flip the image on even or odd indices.",
                        params.system.flipOnEven);

    makeLabeledSpinbox(layout, row++, "Alines Per Bscan", "", "",
                       ioparams.alinesPerBscan, {500, 2000});

    makeLabeledSpinbox(layout, row++, "Rotation offset", "", " lines",
                       params.system.rotateOffset, {-500, 500});

    makeLabeledSpinbox(layout, row++, "RF size (PA)",
                       "Samples per Ascan for PA can be changed here. Samples "
                       "per Ascan for US will be double this.",
                       " pts", ioparams.rfSizePA, {2500, 3000});

    makeLabeledSpinbox(layout, row++, "OffsetUS",
                       "Change this (in no. of samples) to move how close the "
                       "US signals are in relation to the axis of rotation.",
                       " pts", ioparams.offsetUS, {-2000, 2000});

    makeLabeledSpinbox(
        layout, row++, "OffsetPA",
        "Change this (in no. of samples) to coregister PA and US.", " pts",
        ioparams.offsetPA, {-2000, 2000});

    makeLabeledDoubleSpinbox(layout, row++, "SAFT delay multiplier", "", "",
                             params.system.saftTimeDelayMultiplier, {0.1, 10.0},
                             0.1);
  }

  const QString &help_Truncate = "Truncate num points from the beginning to "
                                 "remove pulser/laser artifacts.";
  const QString &help_NoiseFloor =
      "Noise floor (mV) is the maximum noise level that will be cut out.";
  const QString &help_DynamicRange =
      "Dynamic range above the noisefloor that will be displayed.";
  const QString &help_SAFT = "Use SAFT";

  const auto makeReconParamsControl = [&](const QString &groupBoxName,
                                          uspam::recon::ReconParams &p) {
    auto *gb = new CollapsibleGroupBox(groupBoxName);
    auto *vlayout = new QVBoxLayout;
    gb->setLayout(vlayout);

    {
      auto *layout = new QGridLayout;
      vlayout->addLayout(layout);
      int row = 0;

      makeLabeledCheckbox(layout, row++, "Background subtract", "",
                          p.backgroundSubtract);

      // Filter type and order control
      {
        auto *label = new QLabel("Filter type");
        label->setToolTip("select the filter type");
        layout->addWidget(label, row, 0);

        auto *filterTypeCBox = new QComboBox;
        layout->addWidget(filterTypeCBox, row++, 1);

        const auto [firTapsLabel, firTapsSpinBox] =
            makeLabeledSpinbox(layout, row++, "FIR Taps", "FIR num taps", "",
                               p.firTaps, {3, 125}, 2);

        const auto [iirOrderLabel, iirOrderSpinBox] =
            makeLabeledSpinbox(layout, row++, "IIR Order", "IIR filter order",
                               "", p.iirOrder, {1, 25});

        {
          using uspam::recon::FilterType;
          filterTypeCBox->addItem("FIR", QVariant::fromValue(FilterType::FIR));
          filterTypeCBox->addItem("IIR", QVariant::fromValue(FilterType::IIR));

          connect(filterTypeCBox,
                  QOverload<int>::of(&QComboBox::currentIndexChanged),
                  [this, &p, filterTypeCBox, firTapsLabel, firTapsSpinBox,
                   iirOrderLabel, iirOrderSpinBox](int index) {
                    p.filterType = qvariant_cast<FilterType>(
                        filterTypeCBox->itemData(index));

                    switch (p.filterType) {
                    case FilterType::FIR: {
                      firTapsLabel->show();
                      firTapsSpinBox->show();
                      iirOrderLabel->hide();
                      iirOrderSpinBox->hide();
                    } break;
                    case FilterType::IIR: {
                      firTapsLabel->hide();
                      firTapsSpinBox->hide();
                      iirOrderLabel->show();
                      iirOrderSpinBox->show();
                    } break;
                    }

                    this->_paramsUpdatedInternal();
                  });

          updateGuiFromParamsCallbacks.emplace_back(
              [filterTypeCBox, firTapsLabel, firTapsSpinBox, iirOrderLabel,
               iirOrderSpinBox, this, &p] {
                for (int i = 0; i < filterTypeCBox->count(); ++i) {
                  if (qvariant_cast<FilterType>(filterTypeCBox->itemData(i)) ==
                      p.filterType) {
                    filterTypeCBox->setCurrentIndex(i);
                  }
                }

                switch (p.filterType) {
                case FilterType::FIR: {
                  firTapsLabel->show();
                  firTapsSpinBox->show();
                  iirOrderLabel->hide();
                  iirOrderSpinBox->hide();
                } break;
                case FilterType::IIR: {
                  firTapsLabel->hide();
                  firTapsSpinBox->hide();
                  iirOrderLabel->show();
                  iirOrderSpinBox->show();
                } break;
                }
              });
        }
      }

      makeLabeledDoubleSpinbox(layout, row++, "Bandpass low",
                               "Bandpass low frequency", "", p.bpLowFreq,
                               {0.F, 1.F}, 0.01F);

      makeLabeledDoubleSpinbox(layout, row++, "Bandpass high",
                               "Bandpass high frequency", "", p.bpHighFreq,
                               {0.F, 1.F}, 0.01F);

      makeLabeledSpinbox(layout, row++, "Truncate", help_Truncate, " pts",
                         p.truncate, {0, 2000});

      makeLabeledDoubleSpinbox(layout, row++, "Noise floor", help_NoiseFloor,
                               " mV", p.noiseFloor_mV, {0.0F, 60.0F}, 1.0F);

      makeLabeledDoubleSpinbox(layout, row++, "Dynamic range",
                               help_DynamicRange, " dB", p.desiredDynamicRange,
                               {10.0F, 70.0F}, 1.0F);

      // Beamformer
      {
        auto *label = new QLabel("Beamformer");
        label->setToolTip(help_DynamicRange);
        layout->addWidget(label, row, 0);
        using uspam::beamformer::BeamformerType;

        auto *cbox = new QComboBox;
        layout->addWidget(cbox, row++, 1);
        cbox->addItem("None", QVariant::fromValue(BeamformerType::NONE));
        cbox->addItem("SAFT", QVariant::fromValue(BeamformerType::SAFT));
        cbox->addItem("SAFT CF", QVariant::fromValue(BeamformerType::SAFT_CF));

        connect(cbox, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [&, cbox](int index) {
                  p.beamformerType =
                      qvariant_cast<BeamformerType>(cbox->itemData(index));
                  this->_paramsUpdatedInternal();
                });

        updateGuiFromParamsCallbacks.emplace_back([this, cbox, &p] {
          for (int i = 0; i < cbox->count(); ++i) {
            if (qvariant_cast<BeamformerType>(cbox->itemData(i)) ==
                p.beamformerType) {
              cbox->setCurrentIndex(i);
            }
          }
        });
      }
    }

    return gb;
  };

  layout->addWidget(makeReconParamsControl("PA", this->params.PA));
  layout->addWidget(makeReconParamsControl("US", this->params.US));

  layout->addStretch();

  updateGuiFromParams();
}

void ReconParamsController::resetParams2024v1() {
  params = uspam::recon::ReconParams2::system2024v1();
  ioparams = uspam::io::IOParams::system2024v1();
  updateGuiFromParams();
}

void ReconParamsController::resetParams2024v2GUI() {
  params = uspam::recon::ReconParams2::system2024v2GUI();
  ioparams = uspam::io::IOParams::system2024v2GUI();
  updateGuiFromParams();
}

void ReconParamsController::resetParams2024v2GUIprobe2() {
  params = uspam::recon::ReconParams2::system2024v2GUIprobe2();
  ioparams = uspam::io::IOParams::system2024v2GUI();
  updateGuiFromParams();
}

void ReconParamsController::updateGuiFromParams() {
  for (const auto &func : updateGuiFromParamsCallbacks) {
    func();
  }
}

// NOLINTEND(*-magic-numbers)
