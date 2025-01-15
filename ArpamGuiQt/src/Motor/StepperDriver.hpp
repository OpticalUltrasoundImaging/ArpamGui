#pragma once

#include <span>
#include <vector>

namespace motor {

void generateSquareWave(std::span<double> data, double sampleRate,
                        double pulseRate, double amplitudeLow,
                        double amplitudeHigh, double dutyCycle = 0.5);

class StepperDriver {
public:
  // Initialize the StepperConfig with the sampleRate of the analogue output and
  // stepsPerCycle in the stepper motor
  explicit StepperDriver(double sampleRate, double stepsPerCycle = 1600,
                         double amplitudeLow = 0.0, double amplitudeHigh = 5.0,
                         double dutyCycle = 0.5)
      : m_sampleRate(sampleRate), m_stepsPerCycle(stepsPerCycle),
        m_amplitudeLow(amplitudeLow), m_aplitudeHigh(amplitudeHigh),
        m_dutyCycle(dutyCycle){};

  // Speed in rotations per second
  void setSpeed(double speed);

  // Rotations per second
  void setRotations(double rotations);

  [[nodiscard]] double approxMoveTimeSec() const {
    return m_rotations / m_speed;
  }

  [[nodiscard]] bool needsUpdate() const { return m_needsUpdate; }

  void updateSquareWave();

  [[nodiscard]] const std::vector<double> &squareWave();

private:
  double m_sampleRate;
  double m_stepsPerCycle;
  double m_amplitudeLow;
  double m_aplitudeHigh;
  double m_dutyCycle;

  double m_speed{1};
  double m_rotations{1};
  std::vector<double> m_squareWave; // Motor control signal
  bool m_needsUpdate{true};
};

} // namespace motor
