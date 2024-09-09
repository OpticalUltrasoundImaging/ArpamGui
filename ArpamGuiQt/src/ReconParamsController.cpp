#include "ReconParamsController.hpp"
#include "Common.hpp"
#include "SaftParamsController.hpp"
#include "uspam/beamformer/BeamformerType.hpp"
#include "uspam/beamformer/SAFT.hpp"
#include "uspam/reconParams.hpp"
#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QIntValidator>
#include <QPushButton>
#include <QRegularExpression>
#include <QSpinBox>
#include <QValidator>
#include <QVariant>
#include <Qt>
#include <qspinbox.h>
#include <ranges>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {

class DoubleListValidator : public QValidator {
public:
  explicit DoubleListValidator(QObject *parent = nullptr) : QValidator(parent) {
    // Regex to match a list of doubles separated by commas.
    // It handles optional whitespace around commas and after the last number.
    regex = QRegularExpression("");
    regex.optimize();
  }

  QValidator::State validate(QString &input, int &pos) const override {
    if (input.isEmpty() || input.endsWith(",")) {
      // Allow empty string or trailing comma to facilitate typing
      return QValidator::Intermediate;
    }
    auto match = regex.match(input, pos);
    if (match.hasMatch()) {
      return QValidator::Acceptable;
    }
    return QValidator::Invalid;
  }

private:
  QRegularExpression regex;
};

template <typename T>
auto vectorToStdString(const std::vector<T> &vec) -> std::string {
  if (vec.empty()) {
    return {};
  }

  std::stringstream ss;
  ss << vec.front();
  for (const auto &val : vec | std::views::drop(1)) {
    ss << ", " << val;
  }
  return ss.str();
}

} // namespace

// NOLINTBEGIN(*-magic-numbers)

ReconParamsController::ReconParamsController(QWidget *parent)
    : QWidget(parent), params(uspam::recon::ReconParams2::system2024v2probe2()),
      ioparams(uspam::io::IOParams::system2024v2()) {
  auto *layout = new QVBoxLayout();
  this->setLayout(layout);

  auto *doubleListValidator = new DoubleListValidator(this);

  const auto makeQSpinBox = [this](const std::pair<int, int> &range,
                                   int singleStep, int &value) {
    auto *spinBox = new QSpinBox;
    spinBox->setSingleStep(singleStep);
    spinBox->setRange(range.first, range.second);
    spinBox->setValue(value);
    connect(spinBox, &QSpinBox::valueChanged, this,
            [this, &value](int newValue) {
              value = newValue;
              this->_paramsUpdatedInternal();
            });

    this->updateGuiFromParamsCallbacks.emplace_back(
        [spinBox, &value] { spinBox->setValue(value); });
    return spinBox;
  };

  // value1 and value2 are sync'ed
  const auto makeQSpinBox2 = [this](const std::pair<int, int> &range,
                                    int singleStep, int &value1, int &value2) {
    auto *spinBox = new QSpinBox;
    spinBox->setSingleStep(singleStep);
    spinBox->setRange(range.first, range.second);
    spinBox->setValue(value1);
    connect(spinBox, &QSpinBox::valueChanged, this,
            [this, &value1, &value2](int newValue) {
              value1 = newValue;
              value2 = newValue;
              this->_paramsUpdatedInternal();
            });

    this->updateGuiFromParamsCallbacks.emplace_back(
        [spinBox, &value1] { spinBox->setValue(value1); });
    return spinBox;
  };

  const auto makeQDoubleSpinBox =
      [this]<typename Float>(const std::pair<Float, Float> &range,
                             Float singleStep, Float &value) {
        auto *spinBox = new QDoubleSpinBox;
        spinBox->setRange(static_cast<double>(range.first),
                          static_cast<double>(range.second));
        spinBox->setValue(static_cast<double>(value));
        spinBox->setSingleStep(static_cast<double>(singleStep));
        connect(spinBox, &QDoubleSpinBox::valueChanged, this,
                [this, &value](double newValue) {
                  value = static_cast<Float>(newValue);
                  this->_paramsUpdatedInternal();
                });

        this->updateGuiFromParamsCallbacks.emplace_back(
            [spinBox, &value] { spinBox->setValue(value); });
        return spinBox;
      };

  const auto makeBoolCheckBox = [this](bool &value) {
    auto *cb = new QCheckBox();
    using Qt::CheckState;
    connect(cb, &QCheckBox::checkStateChanged, this,
            [this, &value](CheckState state) {
              const auto checked = state == CheckState::Checked;
              value = checked;
              this->_paramsUpdatedInternal();
            });

    updateGuiFromParamsCallbacks.emplace_back([this, cb, &value] {
      cb->setCheckState(value ? Qt::CheckState::Checked
                              : Qt::CheckState::Unchecked);
    });
    return cb;
  };
  // value1 and value2 should be sync'ed
  const auto makeBoolCheckBox2 = [this](bool &value1, bool &value2) {
    auto *cb = new QCheckBox();
    using Qt::CheckState;
    connect(cb, &QCheckBox::checkStateChanged, this,
            [this, &value1, &value2](CheckState state) {
              const auto checked = state == CheckState::Checked;
              value1 = checked;
              value2 = checked;
              this->_paramsUpdatedInternal();
            });

    updateGuiFromParamsCallbacks.emplace_back([this, cb, &value1] {
      cb->setCheckState(value1 ? Qt::CheckState::Checked
                               : Qt::CheckState::Unchecked);
    });
    return cb;
  };

  const QString &help_Truncate = "Truncate num points from the beginning to "
                                 "remove pulser/laser artifacts.";
  const QString &help_NoiseFloor =
      "Noise floor (mV) is the maximum noise level that will be cut out.";
  const QString &help_DynamicRange =
      "Dynamic range above the noisefloor that will be displayed.";
  const QString &help_SAFT = "Use SAFT";

  const auto makeReconParamsControl = [&](const QString &groupBoxName,
                                          uspam::recon::ReconParams &p) {
    auto *gb = new QGroupBox(groupBoxName);
    auto *vlayout = new QVBoxLayout;
    gb->setLayout(vlayout);

    {
      auto *layout = new QGridLayout;
      vlayout->addLayout(layout);
      int row = 1;

      // Background subtraction
      {
        auto *label = new QLabel("Subtract background");
        layout->addWidget(label, row, 0);

        auto *cb = makeBoolCheckBox(p.subtractBackground);
        layout->addWidget(cb, row++, 1);
      }

      // Smoothing filter
      {
        auto *label = new QLabel("Smoothing filter");
        label->setToolTip("Smooth the raw RF data with 2D median filter. "
                          "Effective at removing electronic noise");
        layout->addWidget(label, row, 0);

        auto *sp = makeQSpinBox({1, 7}, 2, p.medfiltKsize);
        layout->addWidget(sp, row++, 1);
      }

      // Filter type and order control
      {
        auto *label = new QLabel("Bandpass Filter");
        label->setToolTip("Select the filter type");
        layout->addWidget(label, row, 0);

        auto *filterTypeCBox = new QComboBox;
        layout->addWidget(filterTypeCBox, row++, 1);

        auto *firTapsLabel = new QLabel("FIR Taps");
        firTapsLabel->setToolTip("FIR num taps");
        layout->addWidget(firTapsLabel, row, 0);

        QSpinBox *firTapsSpinBox{};
        {
          const auto makeQSpinBox = [this](const std::pair<int, int> &range,
                                           int &value, auto *context) {
            auto *spinBox = new QSpinBox;
            spinBox->setRange(range.first, range.second);
            spinBox->setValue(value);
            connect(spinBox, &QSpinBox::valueChanged, context,
                    [this, spinBox, &value](int newValue) {
                      if (newValue % 2 == 0) {
                        value = newValue + 1;
                        spinBox->setValue(value);
                      } else {
                        value = newValue;
                      }
                      this->_paramsUpdatedInternal();
                    });
            this->updateGuiFromParamsCallbacks.emplace_back(
                [spinBox, &value] { spinBox->setValue(value); });
            return spinBox;
          };

          // Ensure firTaps value is odd
          firTapsSpinBox = makeQSpinBox({3, 125}, p.firTaps, this);
          layout->addWidget(firTapsSpinBox, row++, 1);
        }

        auto *iirOrderLabel = new QLabel("IIR Order");
        iirOrderLabel->setToolTip("IIR filter order");
        layout->addWidget(iirOrderLabel, row, 0);

        auto *iirOrderSpinBox = makeQSpinBox({1, 25}, 1, p.iirOrder);
        layout->addWidget(iirOrderSpinBox, row++, 1);

        {
          using uspam::recon::FilterType;
          filterTypeCBox->addItem("FIR", QVariant::fromValue(FilterType::FIR));
          filterTypeCBox->addItem("IIR", QVariant::fromValue(FilterType::IIR));

          connect(filterTypeCBox,
                  QOverload<int>::of(&QComboBox::currentIndexChanged),
                  [filterTypeCBox, firTapsLabel, firTapsSpinBox, iirOrderLabel,
                   iirOrderSpinBox, this, &p](int index) {
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

      {
        auto *label = new QLabel("Bandpass low");
        label->setToolTip("Bandpass low frequency");
        layout->addWidget(label, row, 0);

        auto *sp = makeQDoubleSpinBox({0.F, 1.F}, 0.01F, p.bpLowFreq);
        layout->addWidget(sp, row++, 1);
      }

      {
        auto *label = new QLabel("Bandpass high");
        label->setToolTip("Bandpass high frequency");
        layout->addWidget(label, row, 0);

        auto *sp = makeQDoubleSpinBox({0.F, 1.F}, 0.01F, p.bpHighFreq);
        layout->addWidget(sp, row++, 1);
      }

      {
        auto *label = new QLabel("Padding (top)");
        label->setToolTip("");
        layout->addWidget(label, row, 0);

        auto *spinBox = makeQSpinBox({0, 2000}, 1, p.padding);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");
      }

      {
        auto *label = new QLabel("Truncate");
        label->setToolTip(help_Truncate);
        layout->addWidget(label, row, 0);

        auto *spinBox = makeQSpinBox({0, 2000}, 1, p.truncate);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");
      }

      {
        auto *label = new QLabel("Noise floor");
        label->setToolTip(help_NoiseFloor);
        layout->addWidget(label, row, 0);

        auto *spinBox =
            makeQDoubleSpinBox({0.0F, 60.0F}, 1.0F, p.noiseFloor_mV);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" mV");
      }

      {
        auto *label = new QLabel("Dynamic range");
        label->setToolTip(help_DynamicRange);
        layout->addWidget(label, row, 0);

        auto *spinBox =
            makeQDoubleSpinBox({10.0F, 70.0F}, 1.0F, p.desiredDynamicRange);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" dB");
      }

      {
        auto *label = new QLabel("Clean surface");
        layout->addWidget(label, row, 0);

        auto *cb = makeBoolCheckBox(p.cleanSurface);
        layout->addWidget(cb, row++, 1);
      }
      {
        auto *label = new QLabel("Additional pts to clean");
        layout->addWidget(label, row, 0);

        auto *sb =
            makeQSpinBox({-1000, 1000}, 1, p.additionalSamplesToCleanSurface);
        layout->addWidget(sb, row++, 1);
      }

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

    auto *btnShowReconParams = new QPushButton("Show SAFT Parameters");
    vlayout->addWidget(btnShowReconParams);

    // Beamformer Params
    {
      SaftParamsController *saftParamsController{};
      if (std::holds_alternative<
              uspam::beamformer::SaftDelayParams<ArpamFloat>>(
              p.beamformerParams)) {
        const auto &params =
            std::get<uspam::beamformer::SaftDelayParams<ArpamFloat>>(
                p.beamformerParams);
        saftParamsController = new SaftParamsController(params);
      } else {
        saftParamsController = new SaftParamsController;
      }

      vlayout->addWidget(saftParamsController);

      connect(btnShowReconParams, &QPushButton::pressed,
              [btnShowReconParams, saftParamsController] {
                if (saftParamsController->isVisible()) {
                  saftParamsController->setVisible(false);
                  btnShowReconParams->setText("Show SAFT Parameters");
                } else {
                  saftParamsController->setVisible(true);
                  btnShowReconParams->setText("Hide SAFT Parameters");
                }
              });
      saftParamsController->hide();

      connect(saftParamsController, &SaftParamsController::paramsUpdated,
              [this, &p](uspam::beamformer::SaftDelayParams<float> params) {
                p.beamformerParams = params;
                this->_paramsUpdatedInternal();
              });
    }

    return gb;
  };

  layout->addWidget(makeReconParamsControl("PA", this->params.PA));
  layout->addWidget(makeReconParamsControl("US", this->params.US));

  // Registration params
  {
    auto *gb = new QGroupBox(tr("Coregistration"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 0;

    {
      auto *label = new QLabel("Flip on even");
      label->setToolTip("Flip the image on even or odd indices.");
      layout->addWidget(label, row, 0);

      auto *cb = makeBoolCheckBox2(params.PA.flipOnEven, params.US.flipOnEven);
      layout->addWidget(cb, row++, 1);
    }

    {
      auto *label = new QLabel("Rotation offset");
      label->setToolTip(
          "Rotation offset in no. of Alines to stablize the display.");
      layout->addWidget(label, row, 0);
      auto *spinBox = makeQSpinBox2({-500, 500}, 1, params.US.rotateOffset,
                                    params.PA.rotateOffset);
      layout->addWidget(spinBox, row++, 1);
      spinBox->setSuffix(" lines");
    }

    {
      auto *label = new QLabel("Alines Per Bscan");
      label->setToolTip("");
      layout->addWidget(label, row, 0);
      auto *spinBox = makeQSpinBox({500, 2000}, 1, ioparams.alinesPerBscan);
      spinBox->setSingleStep(100);
      layout->addWidget(spinBox, row++, 1);
    }

    const auto makeIOParams_controller = [&](const QString &prefix,
                                             uspam::io::IOParams_ &p) {
      {
        auto *label = new QLabel(prefix + " start");
        label->setToolTip("Where to start reading in the combined rf array.");
        layout->addWidget(label, row, 0);
        auto *spinBox = makeQSpinBox({0, 8000}, 1, p.start);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");
      }

      {
        auto *label = new QLabel(prefix + " delay");
        label->setToolTip("How much delay the start point is from the axis.");
        layout->addWidget(label, row, 0);
        auto *spinBox = makeQSpinBox({-2000, 2000}, 1, p.delay);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");
      }

      {
        auto *label = new QLabel(prefix + " size");
        label->setToolTip("Num points to read from start.");
        layout->addWidget(label, row, 0);
        auto *spinBox = makeQSpinBox({1000, 8000}, 1, p.size);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");
      }
    };

    makeIOParams_controller("PA", ioparams.PA);
    makeIOParams_controller("US", ioparams.US);
  }

  // Reset buttons
  {
    auto *gb = new QGroupBox("Presets");
    layout->addWidget(gb);

    auto *_layout = new QVBoxLayout;
    gb->setLayout(_layout);

    {
      auto *btn = new QPushButton("2024v1 (Labview)");
      _layout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v1);
    }

    {
      auto *btn = new QPushButton("2024v2 (ArpamGui) Probe 1");
      _layout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v2probe1);
    }

    {
      auto *btn = new QPushButton("2024v2 (ArpamGui) Probe 2");
      _layout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v2probe2);
    }

    {
      auto *btn = new QPushButton("2024v3 (ArpamGui)");
      _layout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v3);
    }

    {
      auto *btn = new QPushButton("Converted old bin");
      _layout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this, [this] {
        params = uspam::recon::ReconParams2::convertedOldBin();
        ioparams = uspam::io::IOParams::convertedOldBin();
        updateGuiFromParams();
      });
    }
  }

  layout->addStretch();

  updateGuiFromParams();
}

// NOLINTEND(*-magic-numbers)
