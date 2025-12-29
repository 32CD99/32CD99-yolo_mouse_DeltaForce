#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <atomic>

struct RoiInfo {
	int full_w = 0;
	int full_h = 0;
	int roi = 640;
	int ox = 0;
	int oy = 0;
};

class DxgiDup {
public:
	// outputGlobalIndex: enumerate outputs across all adapters (0=first monitor)
	bool init(int outputGlobalIndex = 0);
	void shutdown();

	ID3D11Texture2D* acquire_roi_tex(int roi, RoiInfo& outInfo, UINT timeoutMs = 0);

	IDXGIAdapter1* adapter() const { return m_adapter.Get(); }

	// --- UI debug stats (thread-safe reads) ---
	HRESULT last_acquire_hr() const { return (HRESULT)m_lastAcquireHr.load(std::memory_order_relaxed); }
	int cap_fail_streak() const { return m_capFailStreak.load(std::memory_order_relaxed); }
	int cap_recover_count() const { return m_capRecoverCount.load(std::memory_order_relaxed); }
	int cap_black_streak() const { return m_capBlackStreak.load(std::memory_order_relaxed); }

private:
	Microsoft::WRL::ComPtr<IDXGIAdapter1> m_adapter;
	Microsoft::WRL::ComPtr<IDXGIOutput> m_output;
	Microsoft::WRL::ComPtr<ID3D11Device> m_dev;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_ctx;
	Microsoft::WRL::ComPtr<IDXGIOutputDuplication> m_dup;

	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_roiTex;
	int m_roiAlloc = 0;

	// Reuse last ROI when AcquireNextFrame times out (scheme B)
	bool m_hasRoiContent = false;
	int  m_lastRoiCopied = 0;

	int m_w = 0, m_h = 0;
	int m_outputGlobalIndex = 0;

	bool pick_output(int outputGlobalIndex);
	bool create_device_on_adapter();
	bool create_duplication();
	Microsoft::WRL::ComPtr<ID3D11Texture2D> ensure_roi_tex(int roi);

	// --- UI stats / recovery helpers ---
	std::atomic<long> m_lastAcquireHr{ S_OK };
	std::atomic<int>  m_capFailStreak{ 0 };
	std::atomic<int>  m_capRecoverCount{ 0 };
	std::atomic<int>  m_capBlackStreak{ 0 };
	std::atomic<unsigned long long> m_lastAutoRecoverTick{ 0 };

	int m_probeCounter = 0;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_probeStaging; // 1x1 staging tex for black probe
};
