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

constexpr size_t mSystemInfo_Windows11InitialBuild = 21996;

inline bool mSystemInfo_IsWindows11OrGreater() { return mSystemInfo_IsWindows10OrGreater(mSystemInfo_Windows11InitialBuild); }

bool mSystemInfo_IsWindowsServer();
#endif

struct mSystemInfo
{
  mString cpuDescription;
  size_t numberOfProcessors, pageSize;
  size_t totalPhysicalMemory, physicalMemoryAvailable, totalVirtualMemory, virtualMemoryAvailable, totalPagedMemory, pagedMemoryAvailable;
  mString operatingSystemDescription;
  mString preferredUILanguages;
};

mFUNCTION(mSystemInfo_GetInfo, OUT mSystemInfo *pInfo);
mFUNCTION(mSystemInfo_GetDeviceInfo, OUT mString *pManufacturer, OUT mString *pModel);

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

struct mSystemInfo_ProcessPerformanceSnapshot
{
  uint32_t handleCount;
  double_t ioDataOperationsPerSec;
  double_t ioOtherOperationsPerSec;
  double_t ioReadBytesPerSec;
  double_t ioReadOperationsPerSec;
  double_t ioWriteBytesPerSec;
  double_t ioWriteOperationsPerSec;
  double_t ioDataBytesPerSec;
  double_t ioOtherBytesPerSec;
  double_t pageFaultsPerSec;
  uint64_t pageFileBytes;
  uint64_t pageFileBytesPeak;
  double_t percentPrivilegedTime;
  double_t percentProcessorTime;
  double_t percentUserTime;
  uint32_t poolNonpagedBytes;
  uint32_t poolPagedBytes;
  uint32_t priorityBase;
  uint64_t privateBytes;
  uint32_t threadCount;
  uint64_t virtualBytes;
  uint64_t virtualBytesPeak;
  uint64_t workingSet;
  uint64_t workingSetPeak;

  double_t totalSeconds;
  uint64_t totalTicks;
  double_t containedSeconds;
  uint64_t containedTicks;

  uint64_t totalIODataOperations;
  uint64_t totalIOOtherOperations;
  uint64_t totalIOReadBytes;
  uint64_t totalIOReadOperations;
  uint64_t totalIOWriteBytes;
  uint64_t totalIOWriteOperations;
  uint64_t totalIODataBytes;
  uint64_t totalIOOtherBytes;
  uint32_t totalPageFaults;

  double_t gpuCombinedUtilizationPercentage;
  double_t containedGpuSeconds;

  uint64_t gpuTotalMemory;
  uint64_t gpuDedicatedMemory;
  uint64_t gpuSharedMemory;
};

mFUNCTION(mSystemInfo_GetProcessPerformanceInfo, OUT mSystemInfo_ProcessPerformanceSnapshot *pInfo);

#endif // mSystemInfo_h__
