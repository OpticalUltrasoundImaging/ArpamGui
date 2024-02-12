#pragma once

#include <serial/serial.h>

#include <string>
#include <vector>

namespace arpam::motor {
    enum class MotorCommands {
        STEP = 1,
        TO = 2,
        ECHO = 3,
        ERROR = 99,
    };

    /**
     * Command packet structure
     * (1 byte) Start marker (0xFF)
     * (1 byte) Command ID
     * (1 byte) Total length of packet in bytes, including start and end markers
     * (length - 5 bytes) Data
     * (1 byte) Checksum CRC, calculated over all bytes except the start/end markers and the checksum byte
     * (1 byte) End marker (0xFE)
     *
     * Response format
     * (1 byte) Command ID responding to
     * (variable, optional) ASCII message
     * (2 bytes) CRLF
     */

    const uint8_t START_MARKER = 0xFF;
    const uint8_t END_MARKER = 0xFE;

    // for ARPAM motor controller v0.3.0
    class MotorController {
        MotorController() = default;
        void open();
        void find_correct_port();
        inline void close() { this->port.close(); }
        inline bool is_open() { return this->port.isOpen(); }

        std::string send_cmd(const std::string cmd);
        bool ping();
        int get_pos();
        void to_pos(int pos, int delay_us);
        bool moving();

        serial::Serial port;
        std::string portname;
    };

}

