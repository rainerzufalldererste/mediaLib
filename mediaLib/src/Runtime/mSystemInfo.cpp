#include "mSystemInfo.h"

#include "mFile.h"

#include <thread>

#if defined(mPLATFORM_WINDOWS)
#define INITGUID
#include <ddraw.h>
#undef INITGUID

#pragma comment(lib, "dxguid.lib")

#include <powerbase.h>

#pragma comment(lib, "PowrProf.lib")
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "nA8KFwfRZ/k8kJKWIVprdq69FT6R+JlwSjRIu130ygkUz40kSCx9uX2iuE6WVXzyMzqWqUYAqWeixyGh"
#endif

size_t mSystemInfo_GetCpuThreadCount()
{
  return std::thread::hardware_concurrency();
}

//////////////////////////////////////////////////////////////////////////

#if defined(mPLATFORM_WINDOWS)

#ifndef _WIN32_WINNT_WIN10
enum
{
  _WIN32_WINNT_WIN10 = 0x0A00,
  _WIN32_WINNT_WINTHRESHOLD = _WIN32_WINNT_WIN10
};
#endif

bool mSystemInfo_IsOSVersionOrGreater_Raw(const uint32_t majorVersion, const uint32_t minorVersion, const uint16_t servicePack, const uint32_t buildNumber)
{
  OSVERSIONINFOEXW versionInfo;
  mZeroMemory(&versionInfo, 1);

  versionInfo.dwOSVersionInfoSize = sizeof(versionInfo);

  const uint64_t conditionMask =
    VerSetConditionMask(
      VerSetConditionMask(
        VerSetConditionMask(
          VerSetConditionMask(
            0, VER_MAJORVERSION, VER_GREATER_EQUAL),
          VER_MINORVERSION, VER_GREATER_EQUAL),
        VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL),
      VER_BUILDNUMBER, VER_GREATER_EQUAL);

  versionInfo.dwMajorVersion = majorVersion;
  versionInfo.dwMinorVersion = minorVersion;
  versionInfo.wServicePackMajor = servicePack;
  versionInfo.dwBuildNumber = buildNumber;

  return VerifyVersionInfoW(&versionInfo, VER_MAJORVERSION | VER_MINORVERSION | (VER_SERVICEPACKMAJOR * !!servicePack) | (VER_BUILDNUMBER * !!buildNumber), conditionMask) != FALSE;
}

bool mSystemInfo_IsWindowsXPOrGreater(const uint16_t servicePack /* = 0 */)
{
  return mSystemInfo_IsOSVersionOrGreater_Raw(HIBYTE(_WIN32_WINNT_WINXP), LOBYTE(_WIN32_WINNT_WINXP), servicePack, 0);
}

bool mSystemInfo_IsWindowsVistaOrGreater(const uint16_t servicePack /* = 0 */)
{
  return mSystemInfo_IsOSVersionOrGreater_Raw(HIBYTE(_WIN32_WINNT_VISTA), LOBYTE(_WIN32_WINNT_VISTA), servicePack, 0);
}

bool mSystemInfo_IsWindows7OrGreater(const uint16_t servicePack /* = 0 */)
{
  return mSystemInfo_IsOSVersionOrGreater_Raw(HIBYTE(_WIN32_WINNT_WIN7), LOBYTE(_WIN32_WINNT_WIN7), servicePack, 0);
}

bool mSystemInfo_IsWindows8OrGreater()
{
  return mSystemInfo_IsOSVersionOrGreater_Raw(HIBYTE(_WIN32_WINNT_WIN8), LOBYTE(_WIN32_WINNT_WIN8), 0, 0);
}

bool mSystemInfo_IsWindows8Point1OrGreater()
{
  return mSystemInfo_IsOSVersionOrGreater_Raw(HIBYTE(_WIN32_WINNT_WINBLUE), LOBYTE(_WIN32_WINNT_WINBLUE), 0, 0);
}

bool mSystemInfo_IsWindows10OrGreater(const uint32_t buildNumber /* = 0 */)
{
  return mSystemInfo_IsOSVersionOrGreater_Raw(HIBYTE(_WIN32_WINNT_WINTHRESHOLD), LOBYTE(_WIN32_WINNT_WINTHRESHOLD), 0, buildNumber);
}

#endif

bool mSystemInfo_IsDarkMode()
{
  static bool isDarkMode = false;

#if defined (mPLATFORM_WINDOWS)
  // Not sure if this is a documented way to do this, but this is (was) what Eclipse uses, since the other way to do this would be to use unnamed functions in `uxtheme.dll`. This may or may not break in the future, but there may also be a documented way to do this at some point.
  uint32_t value = 1;

  isDarkMode = (mSUCCEEDED(mRegistry_ReadKey("HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize\\AppsUseLightTheme", &value)) && value == 0);
#endif

  return isDarkMode;
}

void mSystemInfo_SetDPIAware()
{
#if defined(mPLATFORM_WINDOWS)
  SetProcessDPIAware();

  HMODULE User32_dll = LoadLibraryW(L"User32.dll");

  if (User32_dll == INVALID_HANDLE_VALUE || User32_dll == NULL)
    return;

#ifndef DPI_AWARENESS_CONTEXT_UNAWARE
  DECLARE_HANDLE(DPI_AWARENESS_CONTEXT);

  typedef enum DPI_AWARENESS
  {
    DPI_AWARENESS_INVALID = -1,
    DPI_AWARENESS_UNAWARE = 0,
    DPI_AWARENESS_SYSTEM_AWARE = 1,
    DPI_AWARENESS_PER_MONITOR_AWARE = 2
  } DPI_AWARENESS;

#define DPI_AWARENESS_CONTEXT_UNAWARE              ((DPI_AWARENESS_CONTEXT)-1)
#define DPI_AWARENESS_CONTEXT_SYSTEM_AWARE         ((DPI_AWARENESS_CONTEXT)-2)
#define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE    ((DPI_AWARENESS_CONTEXT)-3)
#define DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 ((DPI_AWARENESS_CONTEXT)-4)
#define DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED    ((DPI_AWARENESS_CONTEXT)-5)

#endif
  typedef DPI_AWARENESS_CONTEXT (* SetThreadDpiAwarenessContextFunc)(
    DPI_AWARENESS_CONTEXT dpiContext
  );

  SetThreadDpiAwarenessContextFunc pSetThreadDpiAwarenessContext = reinterpret_cast<SetThreadDpiAwarenessContextFunc>(GetProcAddress(User32_dll, "SetThreadDpiAwarenessContext"));

  if (pSetThreadDpiAwarenessContext == nullptr)
    return;

  pSetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE);
#endif
}

mFUNCTION(mSystemInfo_GetSystemMemory, OUT OPTIONAL size_t *pTotalRamBytes, OUT OPTIONAL size_t *pAvailableRamBytes)
{
  mFUNCTION_SETUP();

#if defined(mPLATFORM_WINDOWS)
  MEMORYSTATUSEX memStatus;
  memStatus.dwLength = sizeof(memStatus);
  mERROR_IF(0 == GlobalMemoryStatusEx(&memStatus), mR_InternalError);

  if (pTotalRamBytes != nullptr)
    *pTotalRamBytes = memStatus.ullTotalPhys;

  if (pAvailableRamBytes != nullptr)
    *pAvailableRamBytes = memStatus.ullAvailPhys;

  mRETURN_SUCCESS();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
}

template <class T>
inline static void mSafeRelease(T **ppT)
{
  if (*ppT)
  {
    (*ppT)->Release();
    *ppT = nullptr;
  }
}

mFUNCTION(mSystemInfo_GetVideoCardInfo, OUT mSystemInfo_VideoCardInfo *pGpuInfo)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGpuInfo == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  HRESULT hr;

  HMODULE ddraw_dll = LoadLibraryW(TEXT("ddraw.dll"));
  mDEFER_CALL(ddraw_dll, FreeLibrary);
  mERROR_IF(ddraw_dll == nullptr, mR_NotSupported);

  typedef HRESULT(DirectDrawCreateFunc)(GUID *lpGUID, LPDIRECTDRAW *lplpDD, IUnknown *pUnkOuter);
  DirectDrawCreateFunc *pDirectDrawCreate = reinterpret_cast<DirectDrawCreateFunc *>(GetProcAddress(ddraw_dll, mSTRINGIFY(DirectDrawCreate)));
  mERROR_IF(pDirectDrawCreate == nullptr, mR_NotSupported);

  IDirectDraw *pDirectDraw = nullptr;
  mDEFER_CALL(&pDirectDraw, mSafeRelease);

  // Create DirectDraw instance.
  mERROR_IF(FAILED(hr = pDirectDrawCreate(nullptr, &pDirectDraw, nullptr)), mR_InternalError);

  IDirectDraw7 *pDirectDraw7 = nullptr;
  mDEFER_CALL(&pDirectDraw7, mSafeRelease);

  // Get IDirectDraw7 Interface.
  mERROR_IF(FAILED(hr = pDirectDraw->QueryInterface(IID_IDirectDraw7, reinterpret_cast<void **>(&pDirectDraw7))), mR_InternalError);

  DDCAPS caps;
  mZeroMemory(&caps);
  caps.dwSize = sizeof(caps);

  // Query Combined Video Memory.
  mERROR_IF(FAILED(hr = pDirectDraw->GetCaps(&caps, NULL)), mR_InternalError);

  DDSCAPS2 surfaceCaps;
  mZeroMemory(&surfaceCaps);
  surfaceCaps.dwCaps = DDSCAPS_VIDEOMEMORY | DDSCAPS_LOCALVIDMEM;

  DWORD dedicatedVideoMemory = 0;

  // Query Dedicated Video Memory.
  mERROR_IF(FAILED(hr = pDirectDraw7->GetAvailableVidMem(&surfaceCaps, &dedicatedVideoMemory, nullptr)), mR_InternalError);

  int64_t sharedVideoMemory = caps.dwVidMemTotal - dedicatedVideoMemory;

  if (sharedVideoMemory < 0)
    sharedVideoMemory = 0;

  DDDEVICEIDENTIFIER2 deviceIdentifier;
  mZeroMemory(&deviceIdentifier);

  mERROR_IF(FAILED(hr = pDirectDraw7->GetDeviceIdentifier(&deviceIdentifier, 0)), mR_InternalError);

  pGpuInfo->dedicatedVideoMemory = dedicatedVideoMemory;
  pGpuInfo->sharedVideoMemory = (size_t)sharedVideoMemory;
  pGpuInfo->totalVideoMemory = (size_t)caps.dwVidMemTotal;
  pGpuInfo->freeVideoMemory = (size_t)caps.dwVidMemFree;
  pGpuInfo->driverVersion = (size_t)deviceIdentifier.liDriverVersion.QuadPart;
  pGpuInfo->vendorId = deviceIdentifier.dwVendorId;
  pGpuInfo->deviceChipsetId = deviceIdentifier.dwDeviceId;
  pGpuInfo->deviceChipsetRevision = deviceIdentifier.dwRevision;
  pGpuInfo->deviceBoardId = deviceIdentifier.dwSubSysId;

  mERROR_CHECK(mString_Create(&pGpuInfo->deviceName, deviceIdentifier.szDescription, mARRAYSIZE(deviceIdentifier.szDescription), pGpuInfo->deviceName.pAllocator));
  mERROR_CHECK(mString_Create(&pGpuInfo->driverInfo, deviceIdentifier.szDriver, mARRAYSIZE(deviceIdentifier.szDriver), pGpuInfo->driverInfo.pAllocator));

  mRETURN_SUCCESS();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
}

mFUNCTION(mSystemInfo_GetCpuBaseFrequency, OUT size_t *pBaseFrequencyHz)
{
  mFUNCTION_SETUP();

  // This looks dodgy, but MSDN states the following:
  // "Note that this structure definition was accidentally omitted from WinNT.h. This error will be corrected in the future. In the meantime, to compile your application, include the structure definition contained in this topic in your source code." (https://docs.microsoft.com/de-de/windows/win32/power/processor-power-information-str?redirectedfrom=MSDN, March 2021)
  struct PROCESSOR_POWER_INFORMATION
  {
    ULONG Number;
    ULONG MaxMhz;
    ULONG CurrentMhz;
    ULONG MhzLimit;
    ULONG MaxIdleState;
    ULONG CurrentIdleState;
  } powerInformation[256];

  // I don't want to dynamically allocate memory for the CPU cores. If you have more than 256, sorry, you're not supported atm.
  // Btw: This seems reasonable as `PROCESSOR_NUMBER::Number` is an 8 bit value anyways.
  mERROR_IF(mSystemInfo_GetCpuThreadCount() > mARRAYSIZE(powerInformation), mR_InternalError);

  mERROR_IF(0 != CallNtPowerInformation(ProcessorInformation, nullptr, 0, powerInformation, sizeof(powerInformation)), mR_InternalError);

  PROCESSOR_NUMBER processorNumber;
  GetCurrentProcessorNumberEx(&processorNumber);

  if (pBaseFrequencyHz != nullptr)
    *pBaseFrequencyHz = (size_t)powerInformation[processorNumber.Number].MaxMhz * 1000000ULL;

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemInfo_GetDisplayCount, OUT size_t *pDisplayCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDisplayCount == nullptr, mR_ArgumentNull);

  struct _internal
  {
    static BOOL CALLBACK IncrementMonitors(HMONITOR, HDC, LPRECT, LPARAM pParam)
    {
      size_t *pCount = reinterpret_cast<size_t*>(pParam);
      (*pCount)++;

      return TRUE;
    }
  };

  size_t displayCount = 0;
  mERROR_IF(0 != EnumDisplayMonitors(NULL, NULL, _internal::IncrementMonitors, reinterpret_cast<LPARAM>(&displayCount)), mR_InternalError);

  *pDisplayCount = displayCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mSystemInfo_GetDisplayBounds, const size_t displayIndex, OUT mRectangle2D<int64_t> *pDisplayBounds)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDisplayBounds == nullptr, mR_ArgumentNull);

  struct _internal
  {
    size_t displayIndex = 0;
    bool failed = true;

    size_t requestedDisplayIndex;
    mRectangle2D<int64_t> bounds;

    static BOOL CALLBACK IncrementMonitors(HMONITOR, HDC, LPRECT pRect, LPARAM pParam)
    {
      _internal *pData = reinterpret_cast<_internal *>(pParam);

      if (pData->requestedDisplayIndex == pData->displayIndex)
      {
        if (pRect != nullptr)
        {
          pData->failed = false;
          pData->bounds = mRectangle2D<int64_t>((size_t)pRect->left, (size_t)pRect->top, (size_t)pRect->right - (size_t)pRect->left, (size_t)pRect->bottom - (size_t)pRect->top);
        }
      }

      pData->displayIndex++;

      return TRUE;
    }
  } data;

  data.requestedDisplayIndex = displayIndex;

  mERROR_IF(0 == EnumDisplayMonitors(NULL, NULL, _internal::IncrementMonitors, reinterpret_cast<LPARAM>(&data)), mR_InternalError);
  mERROR_IF(data.failed, mR_InternalError);

  *pDisplayBounds = data.bounds;

  mRETURN_SUCCESS();
}
