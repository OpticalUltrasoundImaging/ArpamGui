#include "ReconParamsController.hpp"
#include <QIntValidator>
#include <QRegularExpression>
#include <QSpinBox>
#include <QValidator>
#include <ranges>
#include <sstream>
#include <string>

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
  auto *layout = new QHBoxLayout();
  this->setLayout(layout);

  auto *doubleListValidator = new DoubleListValidator(this);

  const auto makeQSpinBox = [this](const std::pair<int, int> &range, int &value,
                                   auto *context) {
    auto spinBox = new QSpinBox;
    spinBox->setRange(range.first, range.second);
    spinBox->setValue(value);
    connect(spinBox, &QSpinBox::valueChanged, context, [&](int newValue) {
      value = newValue;
      this->_paramsUpdatedInternal();
    });
    return spinBox;
  };

  // PA params
  {
    auto *gb = new QGroupBox(tr("PA recon params"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 1;

    {
      layout->addWidget(new QLabel("Freq"), row, 1);
      auto *filtFreq = new QLineEdit();
      filtFreq->setValidator(doubleListValidator);
      layout->addWidget(filtFreq, row, 2);
      filtFreq->setReadOnly(true);
      row++;

      filtFreq->setText(
          QString::fromStdString(vectorToStdString(params.filterFreqPA)));
    }

    {
      layout->addWidget(new QLabel("Gain"), row, 1);
      auto *filtGain = new QLineEdit();
      filtGain->setValidator(doubleListValidator);
      layout->addWidget(filtGain, row, 2);
      filtGain->setReadOnly(true);
      row++;

      filtGain->setText(
          QString::fromStdString(vectorToStdString(params.filterGainPA)));
    }

    {
      layout->addWidget(new QLabel("Noise floor"), row, 1);
      auto *spinBox = makeQSpinBox({0, 2000}, params.noiseFloorPA, this);
      layout->addWidget(spinBox, row++, 2);
    }

    {
      layout->addWidget(new QLabel("Dynamic range"), row, 1);
      auto *spinBox =
          makeQSpinBox({10, 70}, params.desiredDynamicRangePA, this);
      layout->addWidget(spinBox, row++, 2);
    }
  }

  // US params
  {
    auto *gb = new QGroupBox(tr("US recon params"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 1;

    {
      layout->addWidget(new QLabel("Freq"), row, 1);
      auto *filtFreq = new QLineEdit();
      filtFreq->setValidator(doubleListValidator);
      layout->addWidget(filtFreq, row, 2);
      filtFreq->setReadOnly(true);
      row++;

      filtFreq->setText(
          QString::fromStdString(vectorToStdString(params.filterFreqUS)));
    }

    {
      layout->addWidget(new QLabel("Gain"), row, 1);
      auto *filtGain = new QLineEdit();
      filtGain->setValidator(doubleListValidator);
      layout->addWidget(filtGain, row, 2);
      filtGain->setReadOnly(true);
      row++;

      filtGain->setText(
          QString::fromStdString(vectorToStdString(params.filterGainUS)));
    }

    {
      layout->addWidget(new QLabel("Noise floor"), row, 1);
      auto *spinBox = makeQSpinBox({0, 2000}, params.noiseFloorUS, this);
      layout->addWidget(spinBox, row++, 2);
    }

    {
      layout->addWidget(new QLabel("Dynamic range"), row, 1);
      auto *spinBox =
          makeQSpinBox({10, 70}, params.desiredDynamicRangeUS, this);
      layout->addWidget(spinBox, row++, 2);
    }
  }

  // Registration params
  {
    auto *gb = new QGroupBox(tr("Registration params"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 0;

    {
      layout->addWidget(new QLabel("Rotation offset"), row, 0);
      auto *spinBox =
          makeQSpinBox({-500, 500}, params.alineRotationOffset, this);
      layout->addWidget(spinBox, row++, 1);
    }

    {
      layout->addWidget(new QLabel("PAUS spacer"), row, 0);
      auto *spinBox = makeQSpinBox({0, 200}, ioparams.rf_size_spacer, this);
      layout->addWidget(spinBox, row++, 1);
    }

    {
      layout->addWidget(new QLabel("OffsetUS"), row, 0);
      auto *spinBox = makeQSpinBox({-500, 1000}, ioparams.offsetUS, this);
      layout->addWidget(spinBox, row++, 1);
    }

    {
      layout->addWidget(new QLabel("OffsetPA"), row, 0);
      auto *spinBox = makeQSpinBox({-500, 1000}, ioparams.offsetPA, this);
      layout->addWidget(spinBox, row++, 1);
    }
  }
}

// NOLINTEND(*-magic-numbers)