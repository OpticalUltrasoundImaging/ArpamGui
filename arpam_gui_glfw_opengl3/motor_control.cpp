#include "motor_control.hpp"
#include <cassert>
#include <array>
#include <initializer_list>
#include <concepts>


namespace arpam::motor {
namespace {

    struct CommandBuffer {
        std::array<char, 32> buf;
        size_t size;
    };

    static CommandBuffer cmdbuf{ 0 };

    template<std::same_as<int16_t> ...T>
    const CommandBuffer& makeCommand(const MotorCommands cmd, auto&& ...params)
    {
        constexpr size_t overhead = 5;
        auto& buf = cmdbuf.buf;
        const std::initializer_list<int16_t> params_list{ params... };

        // Clear buffer
        buf.fill(0);

        buf[0] = START_MARKER;
        buf[1] = cmd;
        const size_t nParams = params_list;
        const size_t lengthBytes = overhead + params_list.size() * 2;
        buf[2] = lengthBytes;
        i = 3;

        for (const auto v : params_list) {
            // Little endian, int16
            buf[i] = (uint8_t)(v & 0xFF);
            buf[i+1] = (uint8_t)((v >> 8) & 0xFF);
            i += 2;
        }

        return cmdbuf;
    }


    // Packed commands are 5 bytes, specified by the `Params` struct.
    // The response is in the same format.
    void _send_packed_cmd(serial::Serial& port, const std::uint8_t action, const std::int16_t param1, const std::int16_t param2) {
        _cmd.cmd.action = action;
        _cmd.cmd.param1 = param1;
        _cmd.cmd.param2 = param2;
        port.write(_cmd.buf, sizeof(_cmd.buf));

        // Read response
        const auto nbytes_read = port.read(_cmd.buf, sizeof(_cmd.buf));
        if (nbytes_read == sizeof(_cmd.buf)) {
            // Success
        }
        else {
            // Failure
        }
    }

    // All commands and responses are separated by '\n'
    std::string _send_cmd(serial::Serial& port, const std::string& msg) {
        assert(port.isOpen());
        port.write(msg);
        port.write("\n");
        const auto resp = port.readline();
        return resp;
    }

    std::string _query(serial::Serial& port, const std::string& prop) {
        const auto ressp = _send_cmd(port, prop);
        const auto value_start = prop.size() + 1;
    }

}

    void MotorController::open()
    {
        this->find_correct_port();
    }

    void MotorController::find_correct_port()
    {
        if (port.isOpen()) port.close();

        this->portname.clear();
        for (int i = 0; i < 255; i++) {
            // Check ports from COM0 to COM255

            const std::string _portname = "COM" + std::to_string(i);
            port.setPort(_portname);
            port.setBaudrate(115200);
            port.setParity(serial::parity_none);
            port.setBytesize(serial::bytesize_t::eightbits);
            port.setStopbits(serial::stopbits_one);
            port.setFlowcontrol(serial::flowcontrol_none);
            serial::Timeout timeout{ 10, 100 };
            port.setTimeout(timeout);

            try {
                port.open();
            }
            catch (std::exception& e) {
                // Couldn't open port. Keep trying
                continue;
            }

            const auto resp = this->port.readline();
            if (resp.starts_with("ARPAM")) {
                portname = _portname;
                return;
            }
            // Otherwise close this port and keep trying.
            port.close();
        }

        throw std::exception("Motor control board not connected.");
    }

    std::string MotorController::send_cmd(const std::string cmd)
    {
        return _send_cmd(port, cmd);
    }

    bool MotorController::ping()
    {
        std::array<std::uint8_t, 5> buf = { 5, 0, 0, 0, 0 };
        port.write(buf.data(), buf.size());
        return false;
    }

    int MotorController::get_pos()
    {
        return 0;
    }

}

