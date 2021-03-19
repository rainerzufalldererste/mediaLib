#include "mSystemInfo.h"

#include "mFile.h"

#include <thread>

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
