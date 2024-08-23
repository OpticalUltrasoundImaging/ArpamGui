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
    : QWidget(parent), params(uspam::recon::ReconParams2::system2024v2GUI()),
      ioparams(uspam::io::IOParams::system2024v2GUI()) {
  auto *layout = new QVBoxLayout();
  this->setLayout(layout);

  auto *doubleListValidator = new DoubleListValidator(this);

  const auto makeQSpinBox = [this](const std::pair<int, int> &range,
                                   int singleStep, int &value, auto *context) {
    auto *spinBox = new QSpinBox;
    spinBox->setSingleStep(singleStep);
    spinBox->setRange(range.first, range.second);
    spinBox->setValue(value);
    connect(spinBox, &QSpinBox::valueChanged, context, [&](int newValue) {
      value = newValue;
      this->_paramsUpdatedInternal();
    });
    return spinBox;
  };

  const auto makeQDoubleSpinBox = [this]<typename Float>(
                                      const std::pair<Float, Float> &range,
                                      Float singleStep, Float &value) {
    auto *spinBox = new QDoubleSpinBox;
    spinBox->setRange(static_cast<double>(range.first),
                      static_cast<double>(range.second));
    spinBox->setValue(static_cast<double>(value));
    spinBox->setSingleStep(static_cast<double>(singleStep));
    connect(spinBox, &QDoubleSpinBox::valueChanged, this, [&](double newValue) {
      value = static_cast<Float>(newValue);
      this->_paramsUpdatedInternal();
    });
    return spinBox;
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

      // Smoothing filter
      {
        auto *label = new QLabel("Smoothing filter");
        label->setToolTip("Smooth the raw RF data with 2D median filter");
        layout->addWidget(label, row, 0);

        auto *sp = makeQSpinBox({1, 7}, 2, p.medfiltKsize, this);
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
            return spinBox;
          };

          // Ensure firTaps value is odd
          firTapsSpinBox = makeQSpinBox({3, 125}, p.firTaps, this);
          layout->addWidget(firTapsSpinBox, row++, 1);
          updateGuiFromParamsCallbacks.emplace_back([this, firTapsSpinBox, &p] {
            firTapsSpinBox->setValue(p.firTaps);
          });
        }

        auto *iirOrderLabel = new QLabel("IIR Order");
        iirOrderLabel->setToolTip("IIR filter order");
        layout->addWidget(iirOrderLabel, row, 0);

        auto *iirOrderSpinBox = makeQSpinBox({1, 25}, 1, p.iirOrder, this);
        layout->addWidget(iirOrderSpinBox, row++, 1);
        updateGuiFromParamsCallbacks.emplace_back([this, iirOrderSpinBox, &p] {
          iirOrderSpinBox->setValue(p.iirOrder);
        });

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

        updateGuiFromParamsCallbacks.emplace_back(
            [&p, sp] { sp->setValue(p.bpLowFreq); });
      }

      {
        auto *label = new QLabel("Bandpass high");
        label->setToolTip("Bandpass high frequency");
        layout->addWidget(label, row, 0);

        auto *sp = makeQDoubleSpinBox({0.F, 1.F}, 0.01F, p.bpHighFreq);
        layout->addWidget(sp, row++, 1);

        updateGuiFromParamsCallbacks.emplace_back(
            [&p, sp] { sp->setValue(p.bpHighFreq); });
      }

      {
        auto *label = new QLabel("Padding (top)");
        label->setToolTip("");
        layout->addWidget(label, row, 0);

        auto *spinBox = makeQSpinBox({0, 2000}, 1, p.padding, this);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");

        updateGuiFromParamsCallbacks.emplace_back(
            [this, spinBox, &p] { spinBox->setValue(p.padding); });
      }

      {
        auto *label = new QLabel("Truncate");
        label->setToolTip(help_Truncate);
        layout->addWidget(label, row, 0);

        auto *spinBox = makeQSpinBox({0, 2000}, 1, p.truncate, this);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");

        updateGuiFromParamsCallbacks.emplace_back(
            [this, spinBox, &p] { spinBox->setValue(p.truncate); });
      }

      {
        auto *label = new QLabel("Noise floor");
        label->setToolTip(help_NoiseFloor);
        layout->addWidget(label, row, 0);

        auto *spinBox =
            makeQDoubleSpinBox({0.0F, 60.0F}, 1.0F, p.noiseFloor_mV);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" mV");

        updateGuiFromParamsCallbacks.emplace_back([this, spinBox, &p] {
          spinBox->setValue(static_cast<double>(p.noiseFloor_mV));
        });
      }

      {
        auto *label = new QLabel("Dynamic range");
        label->setToolTip(help_DynamicRange);
        layout->addWidget(label, row, 0);

        auto *spinBox =
            makeQDoubleSpinBox({10.0F, 70.0F}, 1.0F, p.desiredDynamicRange);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" dB");

        updateGuiFromParamsCallbacks.emplace_back(
            [this, spinBox, &p] { spinBox->setValue(p.desiredDynamicRange); });
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

  layout->addWidget(makeReconParamsControl(tr("PA"), this->params.PA));
  layout->addWidget(makeReconParamsControl(tr("US"), this->params.US));

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
      auto *cb = new QCheckBox();
      layout->addWidget(cb, row++, 1);
      using Qt::CheckState;

      connect(cb, &QCheckBox::checkStateChanged, this,
              [this, cb](CheckState state) {
                const auto checked = state == CheckState::Checked;
                this->params.PA.flipOnEven = checked;
                this->params.US.flipOnEven = checked;
                _paramsUpdatedInternal();
              });

      updateGuiFromParamsCallbacks.emplace_back([this, cb] {
        cb->setCheckState(this->params.PA.flipOnEven
                              ? Qt::CheckState::Checked
                              : Qt::CheckState::Unchecked);
      });
    }

    {
      auto *label = new QLabel("Rotation offset");
      label->setToolTip(
          "Rotation offset in no. of Alines to stablize the display.");
      layout->addWidget(label, row, 0);
      auto *spinBox =
          makeQSpinBox({-500, 500}, 1, params.US.rotateOffset, this);
      layout->addWidget(spinBox, row++, 1);
      spinBox->setSuffix(" lines");

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox] {
        spinBox->setValue(this->params.PA.rotateOffset);
        spinBox->setValue(this->params.US.rotateOffset);
      });
    }

    {
      auto *label = new QLabel("Alines Per Bscan");
      label->setToolTip("");
      layout->addWidget(label, row, 0);
      auto *spinBox =
          makeQSpinBox({500, 2000}, 1, ioparams.alinesPerBscan, this);
      spinBox->setSingleStep(100);
      layout->addWidget(spinBox, row++, 1);
      // spinBox->setSuffix("");

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox] {
        spinBox->setValue(this->ioparams.alinesPerBscan);
      });
    }

    const auto makeIOParams_controller = [&](const QString &prefix,
                                             uspam::io::IOParams_ &p) {
      {
        auto *label = new QLabel(prefix + " start");
        label->setToolTip("Where to start reading in the combined rf array.");
        layout->addWidget(label, row, 0);
        auto *spinBox = makeQSpinBox({0, 8000}, 1, p.start, this);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");

        updateGuiFromParamsCallbacks.emplace_back(
            [spinBox, &p] { spinBox->setValue(p.start); });
      }

      {
        auto *label = new QLabel(prefix + " delay");
        label->setToolTip("How much delay the start point is from the axis.");
        layout->addWidget(label, row, 0);
        auto *spinBox = makeQSpinBox({-2000, 2000}, 1, p.delay, this);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");

        updateGuiFromParamsCallbacks.emplace_back(
            [spinBox, &p] { spinBox->setValue(p.delay); });
      }

      {
        auto *label = new QLabel(prefix + " size");
        label->setToolTip("Num points to read from start.");
        layout->addWidget(label, row, 0);
        auto *spinBox = makeQSpinBox({1000, 8000}, 1, p.size, this);
        layout->addWidget(spinBox, row++, 1);
        spinBox->setSuffix(" pts");

        updateGuiFromParamsCallbacks.emplace_back(
            [spinBox, &p] { spinBox->setValue(p.size); });
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
      auto *btn = new QPushButton("2024v2 (ArpamGui)");
      _layout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v2GUI);
    }

    {
      auto *btn = new QPushButton("2024v3 (ArpamGui)");
      _layout->addWidget(btn);
      connect(btn, &QPushButton::pressed, this,
              &ReconParamsController::resetParams2024v3GUI);
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

void ReconParamsController::resetParams2024v3GUI() {
  params = uspam::recon::ReconParams2::system2024v3GUI();
  ioparams = uspam::io::IOParams::system2024v3GUI();
  updateGuiFromParams();
}

void ReconParamsController::updateGuiFromParams() {
  for (const auto &func : updateGuiFromParamsCallbacks) {
    func();
  }
}

// NOLINTEND(*-magic-numbers)
