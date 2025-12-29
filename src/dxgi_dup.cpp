#include "dxgi_dup.h"
#include <iostream>

using Microsoft::WRL::ComPtr;

static ComPtr<ID3D11Texture2D> create_roi_tex(ID3D11Device* dev, int roi)
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = (UINT)roi;
    desc.Height = (UINT)roi;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;

    // ✅ 更稳：至少要让它 RT-bindable（很多机器只靠 SRV 会注册失败）
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;

    desc.CPUAccessFlags = 0;

    // 先用 0；如果仍失败，再试试 D3D11_RESOURCE_MISC_SHARED
    desc.MiscFlags = 0; // 或：D3D11_RESOURCE_MISC_SHARED;

    ComPtr<ID3D11Texture2D> tex;
    HRESULT hr = dev->CreateTexture2D(&desc, nullptr, &tex);
    if (FAILED(hr)) return nullptr;
    return tex;
}

static bool ensure_probe_staging(ID3D11Device* dev, Microsoft::WRL::ComPtr<ID3D11Texture2D>& staging)
{
    if (staging) return true;
    if (!dev) return false;

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = 1;
    desc.Height = 1;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.BindFlags = 0;

    HRESULT hr = dev->CreateTexture2D(&desc, nullptr, staging.ReleaseAndGetAddressOf());
    return SUCCEEDED(hr) && staging;
}

static bool probe_roi_black(ID3D11Device* dev,
    ID3D11DeviceContext* ctx,
    ID3D11Texture2D* roiTex,
    int roi,
    Microsoft::WRL::ComPtr<ID3D11Texture2D>& staging)
{
    if (!dev || !ctx || !roiTex || roi <= 0) return false;
    if (!ensure_probe_staging(dev, staging)) return false;

    const int cx = roi / 2;
    const int cy = roi / 2;

    D3D11_BOX box{};
    box.left = (UINT)cx;
    box.top = (UINT)cy;
    box.right = (UINT)(cx + 1);
    box.bottom = (UINT)(cy + 1);
    box.front = 0;
    box.back = 1;

    ctx->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, roiTex, 0, &box);
    ctx->Flush();

    D3D11_MAPPED_SUBRESOURCE map{};
    HRESULT hr = ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map);
    if (FAILED(hr) || !map.pData) return false;

    const unsigned int pixel = *(const unsigned int*)map.pData;
    ctx->Unmap(staging.Get(), 0);

    const unsigned char b = (unsigned char)(pixel & 0xFFu);
    const unsigned char g = (unsigned char)((pixel >> 8) & 0xFFu);
    const unsigned char r = (unsigned char)((pixel >> 16) & 0xFFu);

    const int thresh = 2; // near black
    return (r <= thresh && g <= thresh && b <= thresh);
}
bool DxgiDup::pick_output(int outputGlobalIndex)
{
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory))) return false;

    int idx = 0;
    for (UINT a = 0; ; a++) {
        ComPtr<IDXGIAdapter1> adp;
        if (factory->EnumAdapters1(a, &adp) == DXGI_ERROR_NOT_FOUND) break;

        for (UINT o = 0; ; o++) {
            ComPtr<IDXGIOutput> out;
            if (adp->EnumOutputs(o, &out) == DXGI_ERROR_NOT_FOUND) break;

            if (idx == outputGlobalIndex) {
                m_adapter = adp;
                m_output = out;

                DXGI_ADAPTER_DESC1 desc{};
                adp->GetDesc1(&desc);

                char name[256]{};
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name), nullptr, nullptr);
                std::cout << "[dup] picked adapter=" << name << " output=" << outputGlobalIndex << "\n";
                return true;
            }
            idx++;
        }
    }
    return false;
}

bool DxgiDup::create_device_on_adapter()
{
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL fl;
    HRESULT hr = D3D11CreateDevice(
        m_adapter.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        flags,
        nullptr, 0,
        D3D11_SDK_VERSION,
        &m_dev, &fl, &m_ctx);

    return SUCCEEDED(hr);
}

bool DxgiDup::create_duplication()
{
    if (!m_output || !m_dev) return false;

    ComPtr<IDXGIOutput1> out1;
    if (FAILED(m_output.As(&out1))) return false;

    HRESULT hr = out1->DuplicateOutput(m_dev.Get(), &m_dup);
    if (FAILED(hr)) return false;

    DXGI_OUTDUPL_DESC dd{};
    m_dup->GetDesc(&dd);
    m_w = (int)dd.ModeDesc.Width;
    m_h = (int)dd.ModeDesc.Height;

    return true;
}

bool DxgiDup::init(int outputGlobalIndex)
{
    shutdown();
    m_outputGlobalIndex = outputGlobalIndex;

    if (!pick_output(outputGlobalIndex)) {
        std::cerr << "[dup] pick_output failed\n";
        return false;
    }
    if (!create_device_on_adapter()) {
        std::cerr << "[dup] create_device_on_adapter failed\n";
        return false;
    }
    if (!create_duplication()) {
        std::cerr << "[dup] create_duplication failed\n";
        return false;
    }

    std::cout << "[dup] init ok size=" << m_w << "x" << m_h << "\n";
    return true;
}

void DxgiDup::shutdown()
{
    m_roiTex.Reset();
    m_dup.Reset();
    m_ctx.Reset();
    m_dev.Reset();
    m_output.Reset();
    m_adapter.Reset();
    m_roiAlloc = 0;
    m_w = m_h = 0;
}

ComPtr<ID3D11Texture2D> DxgiDup::ensure_roi_tex(int roi)
{
    roi = (roi < 64) ? 64 : roi;
    roi = (roi > 2048) ? 2048 : roi;

    if (!m_roiTex || m_roiAlloc != roi) {
        m_roiTex = create_roi_tex(m_dev.Get(), roi);
        m_roiAlloc = roi;
        // New texture => previous content is invalid
        m_hasRoiContent = false;
        m_lastRoiCopied = 0;
        if (!m_roiTex) {
            std::cerr << "[dup] create roi tex failed\n";
            return nullptr;
        }
    }
    return m_roiTex;
}

ID3D11Texture2D* DxgiDup::acquire_roi_tex(int roi, RoiInfo& outInfo, UINT timeoutMs)
{
    if (!m_dup) return nullptr;

    DXGI_OUTDUPL_FRAME_INFO fi{};
    ComPtr<IDXGIResource> res;

    const HRESULT hr = m_dup->AcquireNextFrame(timeoutMs, &fi, &res);
    m_lastAcquireHr.store((long)hr, std::memory_order_relaxed);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        // Scheme B: never block. If we already have a valid ROI texture from a previous frame,
        // reuse it for inference (input stays "latest known frame").
        auto roiTex = ensure_roi_tex(roi);
        if (!roiTex) return nullptr;

        if (!m_hasRoiContent || m_lastRoiCopied != roi) return nullptr; // not yet primed

        const int ox = (m_w - roi) / 2;
        const int oy = (m_h - roi) / 2;

        outInfo.full_w = m_w;
        outInfo.full_h = m_h;
        outInfo.roi = roi;
        outInfo.ox = ox;
        outInfo.oy = oy;

        ID3D11Texture2D* ret = roiTex.Get();
        ret->AddRef();
        return ret;
    }

    if (hr == DXGI_ERROR_ACCESS_LOST) {
        // Common when switching fullscreen/app permissions/display changes.
        m_capFailStreak.fetch_add(1, std::memory_order_relaxed);

        std::cerr << "[dup] access lost; recreating duplication\n";
        m_dup.Reset();
        if (create_duplication()) {
            m_capRecoverCount.fetch_add(1, std::memory_order_relaxed);
            m_capFailStreak.store(0, std::memory_order_relaxed);
            m_capBlackStreak.store(0, std::memory_order_relaxed);
        }
        return nullptr;
    }

    if (FAILED(hr)) {
        const int streak = m_capFailStreak.fetch_add(1, std::memory_order_relaxed) + 1;
        std::cerr << "[dup] AcquireNextFrame failed hr=0x" << std::hex << hr << std::dec << "\n";

        // Auto recover on long failure streak with cooldown.
        const ULONGLONG nowTick = GetTickCount64();
        const ULONGLONG lastTick = (ULONGLONG)m_lastAutoRecoverTick.load(std::memory_order_relaxed);
        if (streak >= 60 && (nowTick - lastTick) >= 2000) {
            m_lastAutoRecoverTick.store((unsigned long long)nowTick, std::memory_order_relaxed);

            std::cerr << "[dup] auto-recover: recreating duplication after fail streak=" << streak << "\n";
            m_dup.Reset();
            if (create_duplication()) {
                m_capRecoverCount.fetch_add(1, std::memory_order_relaxed);
                m_capFailStreak.store(0, std::memory_order_relaxed);
                m_capBlackStreak.store(0, std::memory_order_relaxed);
            }
        }
        return nullptr;
    }

    // Success
    ComPtr<ID3D11Texture2D> frame;
    res.As(&frame);

    auto roiTex = ensure_roi_tex(roi);
    if (!roiTex) {
        m_dup->ReleaseFrame();
        return nullptr;
    }

    const int ox = (m_w - roi) / 2;
    const int oy = (m_h - roi) / 2;

    D3D11_BOX box{};
    box.left = (UINT)ox;
    box.top = (UINT)oy;
    box.right = (UINT)(ox + roi);
    box.bottom = (UINT)(oy + roi);
    box.front = 0;
    box.back = 1;

    m_ctx->CopySubresourceRegion(roiTex.Get(), 0, 0, 0, 0, frame.Get(), 0, &box);
    // Flush() here is usually unnecessary; CUDA interop map will enforce needed sync.
    m_dup->ReleaseFrame();

    m_hasRoiContent = true;
    m_lastRoiCopied = roi;

    m_lastAcquireHr.store((long)S_OK, std::memory_order_relaxed);
    m_capFailStreak.store(0, std::memory_order_relaxed);

    // Black-screen probe every 8 successful frames (cheap 1x1 readback).
    ++m_probeCounter;
    if ((m_probeCounter & 7) == 0) {
        const bool isBlack = probe_roi_black(m_dev.Get(), m_ctx.Get(), roiTex.Get(), roi, m_probeStaging);
        if (isBlack) {
            const int bs = m_capBlackStreak.fetch_add(1, std::memory_order_relaxed) + 1;

            const ULONGLONG nowTick = GetTickCount64();
            const ULONGLONG lastTick = (ULONGLONG)m_lastAutoRecoverTick.load(std::memory_order_relaxed);
            if (bs >= 30 && (nowTick - lastTick) >= 2000) {
                m_lastAutoRecoverTick.store((unsigned long long)nowTick, std::memory_order_relaxed);

                std::cerr << "[dup] BLACK-RECOVER: suspected black screen; recreating duplication\n";
                m_dup.Reset();
                if (create_duplication()) {
                    m_capRecoverCount.fetch_add(1, std::memory_order_relaxed);
                    m_capBlackStreak.store(0, std::memory_order_relaxed);
                }
                return nullptr;
            }
        }
        else {
            m_capBlackStreak.store(0, std::memory_order_relaxed);
        }
    }

    outInfo.full_w = m_w;
    outInfo.full_h = m_h;
    outInfo.roi = roi;
    outInfo.ox = ox;
    outInfo.oy = oy;

    ID3D11Texture2D* ret = roiTex.Get();
    ret->AddRef();
    return ret;
}
