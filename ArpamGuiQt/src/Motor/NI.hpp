#pragma once

#ifdef ARPAM_HAS_NI

/*
Impl motor control using the NI-DAQmx 64-bit C SDK

Card: NI PCIe-6251

ARPAM machine had NI-DAQmx 19.0 installed, but the NIDAQmx.lib is 32-bit (while
the rest of the libs are 64 bit). I'm installing NI-DAQmx 2022Q3, which is a
known version that supports NI PCIe-6251. The last version of NI-DAQmx that
supports LabVIEW 2019 is 2022Q4.

Using NI-DAQmx in text based programming
https://www.ni.com/en/support/documentation/supplemental/21/using-ni-daqmx-in-text-based-programming-environments.html

NI-DAQmx C Reference
https://www.ni.com/docs/en-US/bundle/ni-daqmx-c-api-ref/page/cdaqmx/help_file_title.html

Some C examples I found
https://github.com/edavis0/nidaqmx-c-examples


In general, data acquisition programming with DAQmx involves the following
steps:

- Create a Task and Virtual Channels
- Configure the Timing Parameters
- Start the Task
- Perform a Read operation from the DAQ
- Perform a Write operation to the DAQ
- Stop and Clear the Task.


*/

#include <NIDAQmx.h>
#include <QtLogging>
#include <fmt/core.h>
#include <string>
#include <vector>

namespace motor {

std::string getNIDAQInfo();

class MotorNI {
public:
  enum class Direction { ANTICLOCKWISE = 0, CLOCKWISE };

  MotorNI() = default;

  [[nodiscard]] bool setDirection(Direction direction);

  // Async move
  [[nodiscard]] bool prepareMove();
  [[nodiscard]] bool startMoveAsync();
  [[nodiscard]] bool waitUntilMoveEnds();
  void abortMove();

  // Blocking move
  [[nodiscard]] bool moveBlocking();

  [[nodiscard]] bool moveClockwise() {
    return setDirection(Direction::CLOCKWISE) && moveBlocking();
  }

  [[nodiscard]] bool moveAnticlockwise() noexcept {
    return setDirection(Direction::ANTICLOCKWISE) && moveBlocking();
  }

  [[nodiscard]] bool moveClockwiseThenAnticlockwise() noexcept {
    return setDirection(Direction::CLOCKWISE) && moveBlocking() &&
           setDirection(Direction::ANTICLOCKWISE) && moveBlocking();
  }

  [[nodiscard]] const char *errMsg() const noexcept { return &errBuf[0]; }

private:
  std::vector<double> data; // Motor control signal
  void *taskHandle{};
  char errBuf[1024] = {'\0'};
  int32_t ret = 0;
};

} // namespace motor

#endif // ARPAM_HAS_NI