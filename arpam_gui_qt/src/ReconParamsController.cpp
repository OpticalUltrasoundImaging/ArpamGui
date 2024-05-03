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
    regex = QRegularExpression("^\s*-?\d+(\.\d+)?\s*(,\s*-?\d+(\.\d+)?\s*)*$");
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

ReconParamsController::ReconParamsController(QWidget *parent)
    : QWidget(parent), params(uspam::recon::ReconParams2::system2024v1()),
      ioparams(uspam::io::IOParams::system2024v1()) {
  auto *layout = new QVBoxLayout();
  this->setLayout(layout);

  {
    auto *gb = new QGroupBox(tr("PA recon params"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 1;

    auto *doubleListValidator = new DoubleListValidator(this);

    {
      auto *filtFreqLabel = new QLabel("Freq");
      auto *filtFreq = new QLineEdit();
      filtFreq->setValidator(doubleListValidator);
      layout->addWidget(filtFreqLabel, row, 1);
      layout->addWidget(filtFreq, row, 2);
      row++;
    }

    {
      auto *filtGainLabel = new QLabel("Gain");
      auto *filtGain = new QLineEdit();
      filtGain->setValidator(doubleListValidator);
      layout->addWidget(filtGainLabel, row, 1);
      layout->addWidget(filtGain, row, 2);
      row++;
    }

    {
      auto *noiseFloorLabel = new QLabel("Noise floor");
      auto *noiseFloor = new QSpinBox();
      noiseFloor->setRange(0, 2000);
      noiseFloor->setValue(params.noiseFloorPA);
      layout->addWidget(noiseFloorLabel, row, 1);
      layout->addWidget(noiseFloor, row, 2);
      row++;
    }
  }

  {
    auto *gb = new QGroupBox(tr("Registration params"));
    layout->addWidget(gb);
    auto *layout = new QGridLayout;
    gb->setLayout(layout);
    int row = 0;

    {
      auto *rotOffsetLabel = new QLabel("Rotation offset");
      auto *rotOffset = new QSpinBox();
      connect(rotOffset, &QSpinBox::valueChanged, this,
              &ReconParamsController::rotOffset_changed);
      rotOffset->setRange(-500, 500);
      rotOffset->setValue(params.alineRotationOffset);
      layout->addWidget(rotOffsetLabel, row, 0);
      layout->addWidget(rotOffset, row, 1);
      row++;
    }

    {
      auto *spacerLabel = new QLabel("PAUS spacer");
      auto *spacer = new QSpinBox();
      connect(spacer, &QSpinBox::valueChanged, this,
              &ReconParamsController::PAUSspacer_changed);
      spacer->setRange(0, 200);
      spacer->setValue(this->ioparams.rf_size_spacer);
      layout->addWidget(spacerLabel, row, 0);
      layout->addWidget(spacer, row, 1);
      row++;
    }

    {
      auto *offsetUSLabel = new QLabel("OffsetUS");
      auto *offsetUS = new QSpinBox();
      connect(offsetUS, &QSpinBox::valueChanged, this,
              &ReconParamsController::offsetUS_changed);
      offsetUS->setRange(-500, 1000);
      offsetUS->setValue(ioparams.offsetUS);
      layout->addWidget(offsetUSLabel, row, 0);
      layout->addWidget(offsetUS, row, 1);
      row++;
    }

    {
      auto *offsetPALabel = new QLabel("OffsetPA");
      auto *offsetPA = new QSpinBox();
      connect(offsetPA, &QSpinBox::valueChanged, this,
              &ReconParamsController::offsetPA_changed);
      offsetPA->setRange(-500, 1000);
      offsetPA->setValue(ioparams.offsetPA);
      layout->addWidget(offsetPALabel, row, 0);
      layout->addWidget(offsetPA, row, 1);
      row++;
    }
  }
}