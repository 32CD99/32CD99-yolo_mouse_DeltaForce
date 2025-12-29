#include "fa.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>

#include <vector>
#include <cstring>

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")

namespace stm32hid
{
    struct FaController::Impl
    {
        HANDLE h = INVALID_HANDLE_VALUE;
        FaConfig cfg{};
        std::wstring path;

        uint16_t feature_len = 0;   // HIDP_CAPS::FeatureReportByteLength
        uint32_t last_err = 0;

        void set_err(DWORD e) { last_err = (uint32_t)e; }

        void reset()
        {
            if (h != INVALID_HANDLE_VALUE)
            {
                CloseHandle(h);
                h = INVALID_HANDLE_VALUE;
            }
            path.clear();
            feature_len = 0;
            last_err = 0;
        }
    };

    // 注意：这里不涉及 FaController::Impl，所以不会触发 private 访问错误
    static bool GetHidPathByVidPid(std::wstring& outPath, uint16_t vid, uint16_t pid, DWORD& lastErr)
    {
        GUID hidGuid;
        HidD_GetHidGuid(&hidGuid);

        HDEVINFO info = SetupDiGetClassDevsW(&hidGuid, nullptr, nullptr,
            DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
        if (info == INVALID_HANDLE_VALUE)
        {
            lastErr = GetLastError();
            return false;
        }

        SP_DEVICE_INTERFACE_DATA ifData{};
        ifData.cbSize = sizeof(ifData);

        for (DWORD i = 0; SetupDiEnumDeviceInterfaces(info, nullptr, &hidGuid, i, &ifData); ++i)
        {
            DWORD required = 0;
            SetupDiGetDeviceInterfaceDetailW(info, &ifData, nullptr, 0, &required, nullptr);
            if (required == 0) continue;

            std::vector<uint8_t> buf(required);
            auto* detail = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W*>(buf.data());
            detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);

            if (!SetupDiGetDeviceInterfaceDetailW(info, &ifData, detail, required, nullptr, nullptr))
                continue;

            HANDLE h = CreateFileW(detail->DevicePath,
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                nullptr,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                nullptr);
            if (h == INVALID_HANDLE_VALUE) continue;

            HIDD_ATTRIBUTES attr{};
            attr.Size = sizeof(attr);
            BOOL ok = HidD_GetAttributes(h, &attr);
            CloseHandle(h);

            if (!ok) continue;

            if (attr.VendorID == vid && attr.ProductID == pid)
            {
                outPath = detail->DevicePath;
                SetupDiDestroyDeviceInfoList(info);
                lastErr = 0;
                return true;
            }
        }

        SetupDiDestroyDeviceInfoList(info);
        lastErr = ERROR_FILE_NOT_FOUND;
        return false;
    }

    FaController::FaController() : p_(new Impl) {}
    FaController::~FaController()
    {
        close();
        delete p_;
        p_ = nullptr;
    }

    bool FaController::open(const FaConfig& cfg)
    {
        close();
        p_->cfg = cfg;

        DWORD e = 0;
        if (!GetHidPathByVidPid(p_->path, cfg.vid, cfg.pid, e))
        {
            p_->set_err(e);
            return false;
        }

        p_->h = CreateFileW(p_->path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (p_->h == INVALID_HANDLE_VALUE)
        {
            p_->set_err(GetLastError());
            p_->reset();
            return false;
        }

        // 获取 FeatureReportByteLength
        PHIDP_PREPARSED_DATA ppd = nullptr;
        if (!HidD_GetPreparsedData(p_->h, &ppd))
        {
            p_->set_err(GetLastError());
            p_->reset();
            return false;
        }

        HIDP_CAPS caps{};
        NTSTATUS st = HidP_GetCaps(ppd, &caps);
        HidD_FreePreparsedData(ppd);

        if (st != HIDP_STATUS_SUCCESS)
        {
            p_->set_err(ERROR_INVALID_DATA);
            p_->reset();
            return false;
        }

        p_->feature_len = caps.FeatureReportByteLength;
        if (p_->feature_len == 0)
        {
            // 设备没有 Feature Report，就无法用 HidD_SetFeature
            p_->set_err(ERROR_NOT_SUPPORTED);
            p_->reset();
            return false;
        }

        p_->set_err(0);
        return true;
    }

    void FaController::close()
    {
        if (!p_) return;
        p_->reset();
    }

    bool FaController::is_open() const
    {
        return p_ && p_->h != INVALID_HANDLE_VALUE;
    }

    std::wstring FaController::device_path() const
    {
        return p_ ? p_->path : L"";
    }

    uint16_t FaController::feature_report_length() const
    {
        return p_ ? p_->feature_len : 0;
    }

    uint32_t FaController::last_win32_error() const
    {
        return p_ ? p_->last_err : 0;
    }

    bool FaController::send_feature_report(const uint8_t* data, size_t n)
    {
        if (!is_open() || p_->feature_len == 0) return false;
        if (!data || n == 0) return false;

        // HidD_SetFeature 要求 buffer[0] 是 ReportID，长度 = FeatureReportByteLength
        std::vector<uint8_t> report(p_->feature_len, 0x00);

        size_t copyN = (n <= report.size()) ? n : report.size();
        std::memcpy(report.data(), data, copyN);

        BOOL ok = HidD_SetFeature(p_->h, report.data(), (ULONG)report.size());
        if (!ok)
        {
            p_->set_err(GetLastError());
            return false;
        }

        p_->set_err(0);
        return true;
    }

    bool FaController::send_enable(bool en)
    {
        // 固定格式：02 AA 01 (1/0) 55
        uint8_t pkt[5] = {
            p_->cfg.feature_report_id,
            p_->cfg.frame_head,
            0x01,
            (uint8_t)(en ? 1 : 0),
            p_->cfg.frame_tail
        };
        return send_feature_report(pkt, sizeof(pkt));
    }

    bool FaController::send_move_once(int8_t dx, int8_t dy)
    {
        // 固定格式：02 AA dx dy 55
        uint8_t pkt[5] = {
            p_->cfg.feature_report_id,
            p_->cfg.frame_head,
            (uint8_t)dx,
            (uint8_t)dy,
            p_->cfg.frame_tail
        };
        return send_feature_report(pkt, sizeof(pkt));
    }

} // namespace stm32hid
