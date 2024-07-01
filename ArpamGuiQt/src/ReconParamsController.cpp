#include "ReconParamsController.hpp"
#include "uspam/beamformer/beamformer.hpp"
#include "uspam/reconParams.hpp"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
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
    : QWidget(parent), params(uspam::recon::ReconParams2::system2024v1()),
      ioparams(uspam::io::IOParams::system2024v1()) {
  auto *layout = new QVBoxLayout();
  this->setLayout(layout);

  auto *doubleListValidator = new DoubleListValidator(this);

  const auto makeQSpinBox = [this](const std::pair<int, int> &range, int &value,
                                   auto *context) {
    auto *spinBox = new QSpinBox;
    spinBox->setRange(range.first, range.second);
    spinBox->setValue(value);
    connect(spinBox, &QSpinBox::valueChanged, context, [&](int newValue) {
      value = newValue;
      this->_paramsUpdatedInternal();
    });
    return spinBox;
  };

  const auto makeQDoubleSpinBox =
      [this]<typename FloatType>(const std::pair<FloatType, FloatType> &range,
                                 FloatType singleStep, FloatType &value,
                                 auto *context) {
        auto *spinBox = new QDoubleSpinBox;
        spinBox->setRange(static_cast<double>(range.first),
                          static_cast<double>(range.second));
        spinBox->setValue(static_cast<double>(value));
        spinBox->setSingleStep(static_cast<double>(singleStep));
        connect(spinBox, &QDoubleSpinBox::valueChanged, context,
                [&](double newValue) {
                  value = static_cast<FloatType>(newValue);
                  this->_paramsUpdatedInternal();
                });
        return spinBox;
      };

  const QString &help_Freq = "Parameters to the FIR filter.";
  const QString &help_Gain = "Parameters to the FIR filter.";

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
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 1;

    {
      auto *label = new QLabel("Freq");
      label->setToolTip(help_Freq);
      layout->addWidget(label, row, 1);

      auto *filtFreq = new QLineEdit();
      filtFreq->setValidator(doubleListValidator);
      layout->addWidget(filtFreq, row, 2);
      filtFreq->setReadOnly(true);
      row++;

      updateGuiFromParamsCallbacks.emplace_back([&p, filtFreq] {
        filtFreq->setText(
            QString::fromStdString(vectorToStdString(p.filterFreq)));
      });
    }

    {
      auto *label = new QLabel("Gain");
      label->setToolTip(help_Gain);
      layout->addWidget(label, row, 1);

      auto *filtGain = new QLineEdit();
      filtGain->setValidator(doubleListValidator);
      layout->addWidget(filtGain, row, 2);
      filtGain->setReadOnly(true);
      row++;

      updateGuiFromParamsCallbacks.emplace_back([&p, filtGain] {
        filtGain->setText(
            QString::fromStdString(vectorToStdString(p.filterGain)));
      });
    }

    {
      auto *label = new QLabel("Truncate");
      label->setToolTip(help_Truncate);
      layout->addWidget(label, row, 1);

      auto *spinBox = makeQSpinBox({0, 1000}, p.truncate, this);
      layout->addWidget(spinBox, row++, 2);
      spinBox->setSuffix(" pts");

      updateGuiFromParamsCallbacks.emplace_back(
          [this, spinBox, &p] { spinBox->setValue(p.truncate); });
    }

    {
      auto *label = new QLabel("Noise floor");
      label->setToolTip(help_NoiseFloor);
      layout->addWidget(label, row, 1);

      auto *spinBox =
          makeQDoubleSpinBox({0.0F, 60.0F}, 1.0F, p.noiseFloor_mV, this);
      layout->addWidget(spinBox, row++, 2);
      spinBox->setSuffix(" mV");

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox, &p] {
        spinBox->setValue(static_cast<double>(p.noiseFloor_mV));
      });
    }

    {
      auto *label = new QLabel("Dynamic range");
      label->setToolTip(help_DynamicRange);
      layout->addWidget(label, row, 1);

      auto *spinBox =
          makeQDoubleSpinBox({10.0F, 70.0F}, 1.0F, p.desiredDynamicRange, this);
      layout->addWidget(spinBox, row++, 2);
      spinBox->setSuffix(" dB");

      updateGuiFromParamsCallbacks.emplace_back(
          [this, spinBox, &p] { spinBox->setValue(p.desiredDynamicRange); });
    }

    // Beamformer
    {
      auto *label = new QLabel("Beamformer");
      label->setToolTip(help_DynamicRange);
      layout->addWidget(label, row, 1);
      using uspam::beamformer::BeamformerType;

      auto *cbox = new QComboBox;
      layout->addWidget(cbox, row++, 2);
      cbox->addItem("None", QVariant::fromValue(BeamformerType::NONE));
      cbox->addItem("SAFT", QVariant::fromValue(BeamformerType::SAFT));
      cbox->addItem("SAFT CF", QVariant::fromValue(BeamformerType::SAFT_CF));

      QObject::connect(
          cbox, QOverload<int>::of(&QComboBox::currentIndexChanged),
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
      auto *label = new QLabel("Rotation offset");
      label->setToolTip(
          "Rotation offset in no. of Alines to stablize the display.");
      layout->addWidget(label, row, 0);
      auto *spinBox = makeQSpinBox({-500, 500}, params.US.rotateOffset, this);
      layout->addWidget(spinBox, row++, 1);
      spinBox->setSuffix(" lines");

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox] {
        spinBox->setValue(this->params.PA.rotateOffset);
        spinBox->setValue(this->params.US.rotateOffset);
      });
    }

    {
      auto *label = new QLabel("PAUS spacer");
      label->setToolTip("");
      layout->addWidget(label, row, 0);
      auto *spinBox = makeQSpinBox({0, 200}, ioparams.rf_size_spacer, this);
      layout->addWidget(spinBox, row++, 1);
      spinBox->setSuffix(" pts");

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox] {
        spinBox->setValue(this->ioparams.rf_size_spacer);
      });
    }

    {
      auto *label = new QLabel("OffsetUS");
      label->setToolTip("Change this (in no. of samples) to move how close the "
                        "US signals are in relation to the axis of rotation.");
      layout->addWidget(label, row, 0);
      auto *spinBox = makeQSpinBox({-2000, 2000}, ioparams.offsetUS, this);
      layout->addWidget(spinBox, row++, 1);
      spinBox->setSuffix(" pts");

      updateGuiFromParamsCallbacks.emplace_back(
          [this, spinBox] { spinBox->setValue(this->ioparams.offsetUS); });
    }

    {
      auto *label = new QLabel("OffsetPA");
      label->setToolTip(
          "Change this (in no. of samples) to coregister PA and US.");
      layout->addWidget(label, row, 0);
      auto *spinBox = makeQSpinBox({-2000, 2000}, ioparams.offsetPA, this);
      layout->addWidget(spinBox, row++, 1);
      spinBox->setSuffix(" pts");

      updateGuiFromParamsCallbacks.emplace_back(
          [this, spinBox] { spinBox->setValue(this->ioparams.offsetPA); });
    }
  }

  // Reset buttons
  {
    auto *_layout = new QVBoxLayout;
    layout->addLayout(_layout);

    auto *btn = new QPushButton("Reset params");
    _layout->addWidget(btn);
    connect(btn, &QPushButton::pressed, this,
            &ReconParamsController::resetParams);
  }

  layout->addStretch();

  updateGuiFromParams();
}

void ReconParamsController::resetParams() {
  params = uspam::recon::ReconParams2::system2024v1();
  ioparams = uspam::io::IOParams::system2024v1();
  updateGuiFromParams();
}

void ReconParamsController::updateGuiFromParams() {
  for (const auto &func : updateGuiFromParamsCallbacks) {
    func();
  }
}

// NOLINTEND(*-magic-numbers)
