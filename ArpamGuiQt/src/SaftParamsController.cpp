#include "SaftParamsController.hpp"

SaftParamsController::SaftParamsController(
    uspam::beamformer::SaftDelayParams<T> params)
    : QWidget(nullptr), m_params(params) {

  const auto makeQDoubleSpinBox = [this](const std::pair<double, double> &range,
                                         const double singleStep, T &value) {
    auto *spinBox = new QDoubleSpinBox;
    spinBox->setRange(range.first, range.second);
    spinBox->setSingleStep(singleStep);
    spinBox->setValue(static_cast<double>(value));
    connect(spinBox, &QDoubleSpinBox::valueChanged, this,
            [&](const double newValue) {
              value = static_cast<T>(newValue);
              this->_paramsUpdatedInternal();
            });
    return spinBox;
  };

  // UI
  auto *layout = new QGridLayout;
  layout->setContentsMargins(0, 0, 0, 0);

  this->setLayout(layout);
  int row{0};

  const auto labelledSpinBox =
      [this, &row, layout, &makeQDoubleSpinBox]<typename Float>(
          const QString &text, const std::pair<double, double> &range,
          const double singleStep, Float &value, const QString &suffix = {},
          const QString &tooltip = {}) {
        auto *label = new QLabel(text);
        layout->addWidget(label, row, 0);

        auto *sb = makeQDoubleSpinBox(range, singleStep, value);
        layout->addWidget(sb, row++, 1);
        if (!suffix.isEmpty()) {
          sb->setSuffix(suffix);
        }

        if (!tooltip.isEmpty()) {
          label->setToolTip(tooltip);
          sb->setToolTip(tooltip);
        }

        const auto updateGuiFromParamsCallback = [&value, sb] {
          sb->setValue(static_cast<double>(value));
        };
      };

  labelledSpinBox("Transducer offset", {0.0, 20.0}, 0.1, m_params.rt, " mm",
                  "Distance from axis of rotation to transducer surface");
  labelledSpinBox("Sound speed", {1000.0, 2000.0}, 1.0, m_params.vs, " mm",
                  "Sound speed");

  labelledSpinBox("Focal length", {0.0, 25.0}, 0.1, m_params.f, " mm",
                  "Transducer focal length");

  labelledSpinBox("Diameter", {0.0, 25.0}, 0.1, m_params.d, " mm",
                  "Transducer diameter");

  labelledSpinBox("Illumination Angle", {0.0, 25.0}, 0.1,
                  m_params.illumAngleDeg, " deg", "Illumination diameter");
}
