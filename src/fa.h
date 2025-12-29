#pragma once
#include <cstdint>
#include <string>

namespace stm32hid
{
    struct FaConfig
    {
        uint16_t vid = 0x0483;
        uint16_t pid = 0x5750;

        uint8_t frame_head = 0xAA;
        uint8_t frame_tail = 0x55;

        // 固定用 Report ID 2 下发控制帧
        uint8_t feature_report_id = 0x02;
    };

    class FaController
    {
    public:
        FaController();
        ~FaController();

        FaController(const FaController&) = delete;
        FaController& operator=(const FaController&) = delete;

        bool open(const FaConfig& cfg = FaConfig{});
        void close();
        bool is_open() const;

        std::wstring device_path() const;
        uint16_t feature_report_length() const;
        uint32_t last_win32_error() const;

        // 02 AA 01 (1/0) 55
        bool send_enable(bool en);

        // 02 AA dx dy 55
        bool send_move_once(int8_t dx, int8_t dy);

        // 更底层：发送 Feature Report（buffer[0] 必须是 ReportID）
        bool send_feature_report(const uint8_t* data, size_t n);

    private:
        struct Impl;   // private 嵌套类型
        Impl* p_;
    };
} // namespace stm32hid
