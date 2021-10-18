#ifndef mSystemInfo_h__
#define mSystemInfo_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "hbyZ62HxX3fPrrCVdsxiw7rdG7XaZusUV3TiftHwRsuD3sQGiogHSDIgQODfsCsjzBV/EhlN+Pl+pgXc"
#endif

size_t mSystemInfo_GetCpuThreadCount();

#if defined(mPLATFORM_WINDOWS)
bool mSystemInfo_IsOSVersionOrGreater_Raw(const uint32_t majorVersion, const uint32_t minorVersion, const uint16_t servicePack, const uint32_t buildNumber);

bool mSystemInfo_IsWindowsXPOrGreater(const uint16_t servicePack = 0);
bool mSystemInfo_IsWindowsVistaOrGreater(const uint16_t servicePack = 0);
bool mSystemInfo_IsWindows7OrGreater(const uint16_t servicePack = 0);
bool mSystemInfo_IsWindows8OrGreater();

// This will require the application to contain a manifest stating it's compatible with Windows 10 or Windows 8.1.
bool mSystemInfo_IsWindows8Point1OrGreater();

// Build Number is the actual Build Number, *NOT* the Version.
// This will require the application to contain a manifest stating it's compatible with Windows 10.
bool mSystemInfo_IsWindows10OrGreater(const uint32_t buildNumber = 0);

inline bool mSystemInfo_IsWindows11OrGreater() { return mSystemInfo_IsWindows10OrGreater(21996); }
#endif

bool mSystemInfo_IsDarkMode();

void mSystemInfo_SetDPIAware();

mFUNCTION(mSystemInfo_GetSystemMemory, OUT OPTIONAL size_t *pTotalRamBytes, OUT OPTIONAL size_t *pAvailableRamBytes);
mFUNCTION(mSystemInfo_GetCpuBaseFrequency, OUT size_t *pBaseFrequencyHz);

struct mSystemInfo_VideoCardInfo
{
  size_t dedicatedVideoMemory, sharedVideoMemory, totalVideoMemory, freeVideoMemory, driverVersion;
  uint32_t vendorId, deviceChipsetId, deviceChipsetRevision, deviceBoardId;
  mString deviceName;
  mString driverInfo;
};

mFUNCTION(mSystemInfo_GetVideoCardInfo, OUT mSystemInfo_VideoCardInfo *pGpuInfo);

mFUNCTION(mSystemInfo_GetDisplayCount, OUT size_t *pDisplayCount);
mFUNCTION(mSystemInfo_GetDisplayBounds, const size_t displayIndex, OUT mRectangle2D<int64_t> *pDisplayBounds);

mFUNCTION(mSystemInfo_IsElevated, OUT bool *pIsElevated);

#endif // mSystemInfo_h__
