#include "StepperDriver.hpp"

#include <span>
#include <vector>

namespace motor {

void generateSquareWave(std::span<double> data, double sampleRate,
                        double pulseRate, double amplitudeLow,
                        double amplitudeHigh, double dutyCycle) {
  // Generate square wave
  const auto samples = data.size();
  const auto period = static_cast<int>(sampleRate / pulseRate);
  const auto periodOn = static_cast<int>(0.5 * period);
  int i = 0;
  while (i < samples) {
    int currPeriod = 0;
    for (; i < samples && currPeriod < periodOn; ++i, ++currPeriod) {
      data[i] = amplitudeLow;
    }
    for (; i < samples && currPeriod < period; ++i, ++currPeriod) {
      data[i] = amplitudeHigh;
    }
  }
}

void StepperDriver::setSpeed(double speed) {
  if (m_speed != speed) {
    m_speed = speed;
    m_needsUpdate = true;
  }
}

void StepperDriver::setRotations(double rotations) {
  if (m_rotations != rotations) {
    m_rotations = rotations;
    m_needsUpdate = true;
  }
}

const std::vector<double> &StepperDriver::squareWave() {
  updateSquareWave();
  return m_squareWave;
}

void StepperDriver::updateSquareWave() {
  if (m_needsUpdate) {
    m_needsUpdate = false;
    const double pulseRate = m_stepsPerCycle * m_speed;
    m_squareWave.resize(m_rotations / m_speed * m_sampleRate);
    generateSquareWave(m_squareWave, m_sampleRate, pulseRate, m_amplitudeLow,
                       m_aplitudeHigh, m_dutyCycle);
  }
}

} // namespace motor
