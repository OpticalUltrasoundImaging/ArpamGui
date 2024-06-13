#include "ReconParamsController.hpp"
#include <QIntValidator>
#include <QPushButton>
#include <QRegularExpression>
#include <QSpinBox>
#include <QValidator>
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

  const QString &help_Freq = "Parameters to the FIR filter.";
  const QString &help_Gain = "Parameters to the FIR filter.";

  const QString &help_NoiseFloor =
      "Noise floor is the maximum noise level that will be cut out.";
  const QString &help_DynamicRange =
      "Dynamic range above the noisefloor that will be displayed.";

  // PA params
  {
    auto *gb = new QGroupBox(tr("PA"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 1;

    {
      auto *label = new QLabel("Freq");
      label->setToolTip(help_Freq);
      layout->addWidget(label, row, 1);

      filtFreqPA = new QLineEdit();
      filtFreqPA->setValidator(doubleListValidator);
      layout->addWidget(filtFreqPA, row, 2);
      filtFreqPA->setReadOnly(true);
      row++;

      updateGuiFromParamsCallbacks.emplace_back([this] {
        this->filtFreqPA->setText(QString::fromStdString(
            vectorToStdString(this->params.filterFreqPA)));
      });
    }

    {
      auto *label = new QLabel("Gain");
      label->setToolTip(help_Gain);
      layout->addWidget(label, row, 1);
      filtGainPA = new QLineEdit();
      filtGainPA->setValidator(doubleListValidator);
      layout->addWidget(filtGainPA, row, 2);
      filtGainPA->setReadOnly(true);
      row++;

      updateGuiFromParamsCallbacks.emplace_back([this] {
        this->filtGainPA->setText(QString::fromStdString(
            vectorToStdString(this->params.filterGainPA)));
      });
    }

    {
      auto *label = new QLabel("Noise floor");
      label->setToolTip(help_NoiseFloor);
      layout->addWidget(label, row, 1);
      auto *spinBox = makeQSpinBox({0, 2000}, params.noiseFloorPA, this);
      layout->addWidget(spinBox, row++, 2);

      updateGuiFromParamsCallbacks.emplace_back(
          [this, spinBox] { spinBox->setValue(this->params.noiseFloorPA); });
    }

    {
      auto *label = new QLabel("Dynamic range");
      label->setToolTip(help_DynamicRange);
      layout->addWidget(label, row, 1);
      auto *spinBox =
          makeQSpinBox({10, 70}, params.desiredDynamicRangePA, this);
      layout->addWidget(spinBox, row++, 2);

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox] {
        spinBox->setValue(this->params.desiredDynamicRangePA);
      });
    }
  }

  // US params
  {
    auto *gb = new QGroupBox(tr("US"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 1;

    {
      auto *label = new QLabel("Freq");
      label->setToolTip(help_Freq);
      layout->addWidget(label, row, 1);

      filtFreqUS = new QLineEdit;
      filtFreqUS->setValidator(doubleListValidator);
      layout->addWidget(filtFreqUS, row, 2);
      filtFreqUS->setReadOnly(true);
      row++;

      updateGuiFromParamsCallbacks.emplace_back([this] {
        this->filtFreqUS->setText(QString::fromStdString(
            vectorToStdString(this->params.filterFreqUS)));
      });
    }

    {
      auto *label = new QLabel("Gain");
      label->setToolTip(help_Gain);
      layout->addWidget(label, row, 1);

      filtGainUS = new QLineEdit;
      filtGainUS->setValidator(doubleListValidator);
      layout->addWidget(filtGainUS, row, 2);
      filtGainUS->setReadOnly(true);
      row++;

      updateGuiFromParamsCallbacks.emplace_back([this] {
        this->filtGainUS->setText(QString::fromStdString(
            vectorToStdString(this->params.filterGainUS)));
      });
    }

    {
      auto *label = new QLabel("Noise floor");
      label->setToolTip(help_NoiseFloor);
      layout->addWidget(label, row, 1);
      auto *spinBox = makeQSpinBox({0, 2000}, params.noiseFloorUS, this);
      layout->addWidget(spinBox, row++, 2);

      updateGuiFromParamsCallbacks.emplace_back(
          [this, spinBox] { spinBox->setValue(this->params.noiseFloorUS); });
    }

    {
      auto *label = new QLabel("Dynamic range");
      label->setToolTip(help_DynamicRange);
      layout->addWidget(label, row, 1);
      auto *spinBox =
          makeQSpinBox({10, 70}, params.desiredDynamicRangeUS, this);
      layout->addWidget(spinBox, row++, 2);

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox] {
        spinBox->setValue(this->params.desiredDynamicRangeUS);
      });
    }
  }

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
      auto *spinBox =
          makeQSpinBox({-500, 500}, params.alineRotationOffset, this);
      layout->addWidget(spinBox, row++, 1);

      updateGuiFromParamsCallbacks.emplace_back([this, spinBox] {
        spinBox->setValue(this->params.alineRotationOffset);
      });
    }

    {
      auto *label = new QLabel("PAUS spacer");
      label->setToolTip("");
      layout->addWidget(label, row, 0);
      auto *spinBox = makeQSpinBox({0, 200}, ioparams.rf_size_spacer, this);
      layout->addWidget(spinBox, row++, 1);

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