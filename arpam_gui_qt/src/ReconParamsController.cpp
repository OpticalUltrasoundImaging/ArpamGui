#include "ReconParamsController.hpp"
#include <QIntValidator>
#include <QRegularExpression>
#include <QSpinBox>
#include <QValidator>

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

} // namespace

// NOLINTBEGIN(*-magic-numbers)

ReconParamsController::ReconParamsController(QWidget *parent)
    : QWidget(parent), params(uspam::recon::ReconParams2::system2024v1()),
      ioparams(uspam::io::IOParams::system2024v1()) {
  auto *layout = new QHBoxLayout();
  this->setLayout(layout);

  auto *doubleListValidator = new DoubleListValidator(this);

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
      filtFreq->setDisabled(true);
      row++;
    }

    {
      layout->addWidget(new QLabel("Gain"), row, 1);
      auto *filtGain = new QLineEdit();
      filtGain->setValidator(doubleListValidator);
      layout->addWidget(filtGain, row, 2);
      filtGain->setDisabled(true);
      row++;
    }

    {
      layout->addWidget(new QLabel("Noise floor"), row, 1);
      auto *noiseFloor = new QSpinBox();
      noiseFloor->setRange(0, 2000);
      noiseFloor->setValue(params.noiseFloorPA);
      connect(noiseFloor, &QSpinBox::valueChanged, this,
              &ReconParamsController::noiseFloorPA_changed);
      layout->addWidget(noiseFloor, row, 2);
      row++;
    }

    {
      layout->addWidget(new QLabel("Dynamic range"), row, 1);
      auto *dr = new QSpinBox();
      dr->setRange(10, 70);
      dr->setValue(params.desiredDynamicRangePA);
      connect(dr, &QSpinBox::valueChanged, this,
              &ReconParamsController::dynamicRangePA_changed);
      layout->addWidget(dr, row, 2);
      row++;
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
      filtFreq->setDisabled(true);
      row++;
    }

    {
      layout->addWidget(new QLabel("Gain"), row, 1);
      auto *filtGain = new QLineEdit();
      filtGain->setValidator(doubleListValidator);
      layout->addWidget(filtGain, row, 2);
      filtGain->setDisabled(true);
      row++;
    }

    {
      layout->addWidget(new QLabel("Noise floor"), row, 1);
      auto *noiseFloor = new QSpinBox();
      noiseFloor->setRange(0, 2000);
      noiseFloor->setValue(params.noiseFloorUS);
      connect(noiseFloor, &QSpinBox::valueChanged, this,
              &ReconParamsController::noiseFloorUS_changed);
      layout->addWidget(noiseFloor, row, 2);
      row++;
    }

    {
      layout->addWidget(new QLabel("Dynamic range"), row, 1);
      auto *dr = new QSpinBox();
      dr->setRange(10, 70);
      dr->setValue(params.desiredDynamicRangeUS);
      connect(dr, &QSpinBox::valueChanged, this,
              &ReconParamsController::dynamicRangeUS_changed);
      layout->addWidget(dr, row, 2);
      row++;
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
      auto *rotOffset = new QSpinBox();
      connect(rotOffset, &QSpinBox::valueChanged, this,
              &ReconParamsController::rotOffset_changed);
      rotOffset->setRange(-500, 500);
      rotOffset->setValue(params.alineRotationOffset);
      layout->addWidget(rotOffset, row, 1);
      row++;
    }

    {
      layout->addWidget(new QLabel("PAUS spacer"), row, 0);
      auto *spacer = new QSpinBox();
      connect(spacer, &QSpinBox::valueChanged, this,
              &ReconParamsController::PAUSspacer_changed);
      spacer->setRange(0, 200);
      spacer->setValue(this->ioparams.rf_size_spacer);
      layout->addWidget(spacer, row, 1);
      row++;
    }

    {
      layout->addWidget(new QLabel("OffsetUS"), row, 0);
      auto *offsetUS = new QSpinBox();
      connect(offsetUS, &QSpinBox::valueChanged, this,
              &ReconParamsController::offsetUS_changed);
      offsetUS->setRange(-500, 1000);
      offsetUS->setValue(ioparams.offsetUS);
      layout->addWidget(offsetUS, row, 1);
      row++;
    }

    {
      layout->addWidget(new QLabel("OffsetPA"), row, 0);
      auto *offsetPA = new QSpinBox();
      connect(offsetPA, &QSpinBox::valueChanged, this,
              &ReconParamsController::offsetPA_changed);
      offsetPA->setRange(-500, 1000);
      offsetPA->setValue(ioparams.offsetPA);
      layout->addWidget(offsetPA, row, 1);
      row++;
    }
  }
}

// NOLINTEND(*-magic-numbers)