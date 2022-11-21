#include <windows.h>
#include <winnt.h>
#include <intsafe.h>
#include <cstdio>
#include <chrono>
#include <fcntl.h>
#include <corecrt_io.h>

#include "mediaLib.h"

#include "mProfiler.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "hJxXtXZLn5mrTXicBqpRmqzOi/HGriWHIaPhBoYJdH/uA6vV5/aWehfGSsFZG41NQnPfqNvSkmhJriq2"
#endif

#ifdef mPLATFORM_WINDOWS
HANDLE mStdOutHandle = nullptr;
HANDLE mFileOutHandle = nullptr;
constexpr bool mStdOutVTCodeColours = false; // Disabled by default. Code Path works but is probably slower and adds more clutter.
bool mStdOutForceStdIO = false;
FILE *pStdOut = nullptr;
#endif

inline void mInitConsole()
{
#ifdef mPLATFORM_WINDOWS
  if (mStdOutHandle == nullptr)
  {
    mStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

    if (mStdOutHandle == INVALID_HANDLE_VALUE)
      mStdOutHandle = nullptr;

    if (mStdOutHandle != nullptr)
    {
      DWORD consoleMode = 0;
      mStdOutForceStdIO = !GetConsoleMode(mStdOutHandle, &consoleMode);

      if (mStdOutForceStdIO)
      {
        const int32_t handle = _open_osfhandle(reinterpret_cast<intptr_t>(mStdOutHandle), _O_TEXT);
        
        if (handle != -1)
          pStdOut = _fdopen(handle, "w");
      }

      // I presume VT Colour Codes are in fact slower than `SetConsoleTextAttribute`, since the terminal has to parse those sequences out.
      // Also they're only supported since Windows 10 and would add another code path, therefore they're currently `constexpr` to be `false`.
      //mStdOutVTCodeColours = (!mStdOutForceStdIO && SetConsoleMode(mStdOutHandle, consoleMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING));

      constexpr DWORD codepage_UTF8 = 65001;
      SetConsoleOutputCP(codepage_UTF8);
    }
  }
#endif
}

void mCreateConsole()
{
#ifdef mPLATFORM_WINDOWS
  if (GetConsoleWindow() == nullptr)
    if (0 == AllocConsole())
      return;

  mStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

  if (mStdOutHandle == INVALID_HANDLE_VALUE)
    mStdOutHandle = nullptr;

  if (mStdOutHandle != nullptr)
  {
    DWORD consoleMode = 0;
    mStdOutForceStdIO = !GetConsoleMode(mStdOutHandle, &consoleMode);

    if (mStdOutForceStdIO)
    {
      const int32_t handle = _open_osfhandle(reinterpret_cast<intptr_t>(mStdOutHandle), _O_TEXT);

      if (handle != -1)
        pStdOut = _fdopen(handle, "w");
    }

    // I presume VT Colour Codes are in fact slower than `SetConsoleTextAttribute`, since the terminal has to parse those sequences out.
    // Also they're only supported since Windows 10 and would add another code path, therefore they're currently `constexpr` to be `false`.
    //mStdOutVTCodeColours = (!mStdOutForceStdIO && SetConsoleMode(mStdOutHandle, consoleMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING));

    constexpr DWORD codepage_UTF8 = 65001;
    SetConsoleOutputCP(codepage_UTF8);
  }
#endif
}

#ifdef mPLATFORM_WINDOWS
inline WORD mGetWindowsConsoleColourFromConsoleColour(const mConsoleColour colour)
{
  switch (colour & 0xF)
  {
  default:
  case mCC_Black: return 0;
  case mCC_DarkBlue: return 1;
  case mCC_DarkGreen: return 2;
  case mCC_DarkCyan: return 3;
  case mCC_DarkRed: return 4;
  case mCC_DarkMagenta: return 5;
  case mCC_DarkYellow: return 6;
  case mCC_BrightGray: return 7;
  case mCC_DarkGray: return 8;
  case mCC_BrightBlue: return 9;
  case mCC_BrightGreen: return 10;
  case mCC_BrightCyan: return 11;
  case mCC_BrightRed: return 12;
  case mCC_BrightMagenta: return 13;
  case mCC_BrightYellow: return 14;
  case mCC_White: return 15;
  }
}
#endif

void mResetConsoleColour()
{
  mInitConsole();

#ifdef mPLATFORM_WINDOWS
  if (mStdOutHandle != nullptr)
  {
    if (mStdOutVTCodeColours)
    {
      const char sequence[] = "\x1b[0m";

      WriteConsoleA(mStdOutHandle, sequence, (DWORD)(mARRAYSIZE(sequence) - 1), nullptr, nullptr);
    }
    else if (!mStdOutForceStdIO)
    {
      SetConsoleTextAttribute(mStdOutHandle, mGetWindowsConsoleColourFromConsoleColour(mCC_BrightGray) | (mGetWindowsConsoleColourFromConsoleColour(mCC_Black) << 8));
    }
  }
#else
  fputs("\x1b[0m", stdout);
#endif
}

void mSetConsoleColour(const mConsoleColour foregroundColour, const mConsoleColour backgroundColour)
{
  mInitConsole();

#ifdef mPLATFORM_WINDOWS
  if (mStdOutHandle != nullptr)
  {
    if (mStdOutVTCodeColours)
    {
      const uint8_t fgColour = (foregroundColour & 0xF);
      const uint8_t bgColour = (backgroundColour & 0xF);

      char sequence[64]; // make sure all VT Colour Codes _ALWAYS_ fit inside this buffer.
      mASSERT_DEBUG(mFormat_GetMaxRequiredBytes("\x1b[", (uint8_t)(fgColour < 0x8 ? (30 + fgColour) : (90 - 8 + fgColour)), ";", (uint8_t)(bgColour < 0x8 ? (40 + bgColour) : (100 - 8 + bgColour)), "m") <= mARRAYSIZE(sequence), "Insufficient VT Code Sequence Buffer. This may cause printing errors. Don't do whatever you're doing.");
      const mResult result = mFormatTo(sequence, mARRAYSIZE(sequence), "\x1b[", (uint8_t)(fgColour < 0x8 ? (30 + fgColour) : (90 - 8 + fgColour)), ";", (uint8_t)(bgColour < 0x8 ? (40 + bgColour) : (100 - 8 + bgColour)), "m");
      mASSERT_DEBUG(mSUCCEEDED(result), "Failed to mFormatTo the VT Code Sequence.");

      WriteConsoleA(mStdOutHandle, sequence, (DWORD)strnlen(sequence, mARRAYSIZE(sequence)), nullptr, nullptr);
    }
    else if (!mStdOutForceStdIO)
    {
      const WORD fgColour = mGetWindowsConsoleColourFromConsoleColour(foregroundColour);
      const WORD bgColour = mGetWindowsConsoleColourFromConsoleColour(backgroundColour);
      
      SetConsoleTextAttribute(mStdOutHandle, fgColour | (bgColour << 4));
    }
  }
#else
  const size_t fgColour = (foregroundColour & 0xF);
  const size_t bgColour = (backgroundColour & 0xF);

  printf("\x1b[%" PRIu64 ";%" PRIu64 "m", fgColour < 0x8 ? (30 + fgColour) : (90 - 8 + fgColour), bgColour < 0x8 ? (40 + bgColour) : (100 - 8 + bgColour));
#endif
}

#ifdef mPLATFORM_WINDOWS
void mPrintToOutputWithLength(const char *text, const size_t length)
{
  mInitConsole();

  if (mStdOutHandle != nullptr)
  {
    if (!mStdOutForceStdIO)
      WriteConsoleA(mStdOutHandle, text, (DWORD)length, nullptr, nullptr);
    else if (pStdOut != nullptr)
      fputs(text, pStdOut);
  }

  if (mFileOutHandle != nullptr)
    WriteFile(mFileOutHandle, text, (DWORD)length, nullptr, nullptr);
}
#endif

void mPrintToOutput(const char *text)
{
  if (text == nullptr)
    return;

#ifdef mPLATFORM_WINDOWS
  mPrintToOutputWithLength(text, strlen(text));
#else
  fputs(text, stdout);
#endif
}

void mDefaultPrint(const char *text)
{
  mPrintToOutput(text);

#ifndef GIT_BUILD
  mDebugOut(text);
#endif
}

void mDebugPrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  mPrintToOutputArray("[Debug] ");
  mSetConsoleColour(mCC_DarkGray, mCC_Black);
  mPrintToOutput(text);
  mResetConsoleColour();
  mPrintToOutputArray("\n");

#ifndef GIT_BUILD
  mDebugOut("[Debug] ");
  mDebugOut(text);
  mDebugOut("\n");
#endif
}

void mLogPrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  mPrintToOutputArray("[Log]   ");
  mSetConsoleColour(mCC_DarkCyan, mCC_Black);
  mPrintToOutput(text);
  mResetConsoleColour();
  mPrintToOutputArray("\n");

#ifndef GIT_BUILD
  mDebugOut("[Log]   ");
  mDebugOut(text);
  mDebugOut("\n");
#endif
}

void mErrorPrint(const char *text)
{
  mSetConsoleColour(mCC_BrightRed, mCC_Black);
  mPrintToOutputArray("[Error] ");
  mSetConsoleColour(mCC_BrightYellow, mCC_DarkRed);
  mPrintToOutput(text);
  mResetConsoleColour();
  mPrintToOutputArray("\n");

#ifndef GIT_BUILD
  mDebugOut("[Error] ");
  mDebugOut(text);
  mDebugOut("\n");
#endif
}

void mTracePrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  mPrintToOutputArray("[Trace] ");
  mSetConsoleColour(mCC_BrightGray, mCC_DarkGray);
  mPrintToOutput(text);
  mResetConsoleColour();
  mPrintToOutputArray("\n");

#ifndef GIT_BUILD
  mDebugOut("[Trace] ");
  mDebugOut(text);
  mDebugOut("\n");
#endif
}

void mSilencablePrint(const char *text)
{
  if (!g_mResult_silent && mPrintCallback != nullptr)
    mPrintCallback(text);
}

void mSilencablePrintError(const char *text)
{
  if (!g_mResult_silent && mPrintErrorCallback != nullptr)
    mPrintErrorCallback(text);
}

void mSilencablePrintLog(const char *text)
{
  if (!g_mResult_silent && mPrintLogCallback != nullptr)
    mPrintLogCallback(text);
}

void mSilencablePrintDebug(const char *text)
{
  if (!g_mResult_silent && mPrintDebugCallback != nullptr)
    mPrintDebugCallback(text);
}

void mSilencablePrintTrace(const char *text)
{
  if (!g_mResult_silent && mPrintTraceCallback != nullptr)
    mPrintTraceCallback(text);
}

mResult mSetOutputFilePath(const mString &path, const bool append /* = true */)
{
  mFUNCTION_SETUP();

  wchar_t wpath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(path, wpath, mARRAYSIZE(wpath)));

  HANDLE file = CreateFileW(wpath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  mERROR_IF(file == NULL || file == INVALID_HANDLE_VALUE, mR_IOFailure);
  mDEFER_CALL_ON_ERROR(file, CloseHandle);

  if (GetLastError() == ERROR_ALREADY_EXISTS && append)
    SetFilePointer(file, 0, NULL, FILE_END);

  mResetOutputFile();

  mFileOutHandle = file;

  mRETURN_SUCCESS();
}

void mResetOutputFile()
{
  if (mFileOutHandle == NULL)
    return;

  HANDLE file = mFileOutHandle;
  mFileOutHandle = NULL;

  FlushFileBuffers(file);
  CloseHandle(file);
}

void mFlushOutput()
{
  if (mFileOutHandle != NULL)
    FlushFileBuffers(mFileOutHandle);
}

mPrintCallbackFunc *mPrintCallback = &mDefaultPrint;
mPrintCallbackFunc *mPrintErrorCallback = &mErrorPrint;
mPrintCallbackFunc *mPrintLogCallback = &mLogPrint;
mPrintCallbackFunc *mPrintDebugCallback = &mDebugPrint;
mPrintCallbackFunc *mPrintTraceCallback = &mTracePrint;

//////////////////////////////////////////////////////////////////////////

extern "C" void NTAPI tls_callback(PVOID /* dllHandle */, const DWORD reason, PVOID /* reserved */)
{
  switch (reason)
  {
  case DLL_PROCESS_ATTACH:
    mMemory_OnProcessStart();
    break;

  case DLL_THREAD_ATTACH:
    mMemory_OnThreadStart();
    break;

  case DLL_THREAD_DETACH:
    mMemory_OnThreadExit();
    break;

  case DLL_PROCESS_DETACH:
    mMemory_OnProcessExit();
    break;
  }
}

extern "C"
{
  extern DWORD _tls_used; // The TLS directory (located in .rdata).
}

#pragma section(".CRT$XLY", long, read)
extern "C" __declspec(allocate(".CRT$XLY")) PIMAGE_TLS_CALLBACK _xl_y = tls_callback;

#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:_xl_y")

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSleep, const size_t milliseconds)
{
  mFUNCTION_SETUP();

  mERROR_IF(milliseconds > DWORD_MAX, mR_ArgumentOutOfBounds);

  mPROFILE_SCOPED("mSleep");

  Sleep((DWORD)milliseconds);

  mRETURN_SUCCESS();
}

void mAssert_Internal(const char *expression, const char *text, const char *function, const char *file, const int32_t line)
{
#if defined (_DEBUG)
  _CrtDbgReport(_CRT_ASSERT, file, line, function, "Error: '%s'.\nExpression: %s\n", text, expression);
#else
  mUnused(expression, text, function, file, line);
#endif

  BOOL isRemoteDebuggerPresent = FALSE;

  if (!CheckRemoteDebuggerPresent(GetCurrentProcess(), &isRemoteDebuggerPresent))
    isRemoteDebuggerPresent = FALSE;

  if (isRemoteDebuggerPresent || IsDebuggerPresent())
    __debugbreak();
}

void mFail_Internal(const char *text, const char *function, const char *file, const int32_t line)
{
#if defined (_DEBUG)
  _CrtDbgReport(_CRT_ASSERT, file, line, function, "Error: '%s'.\n", text);
#else
  mUnused(text, function, file, line);
#endif

  BOOL isRemoteDebuggerPresent = FALSE;

  if (!CheckRemoteDebuggerPresent(GetCurrentProcess(), &isRemoteDebuggerPresent))
    isRemoteDebuggerPresent = FALSE;

  if (isRemoteDebuggerPresent || IsDebuggerPresent())
    __debugbreak();
}

int64_t mGetCurrentTimeMs()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int64_t mGetCurrentTimeNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

#ifdef _MSC_VER
#define cpuid __cpuid
#else
#include <cpuid.h>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "hJxXtXZLn5mrTXicBqpRmqzOi/HGriWHIaPhBoYJdH/uA6vV5/aWehfGSsFZG41NQnPfqNvSkmhJriq2"
#endif

void cpuid(int info[4], int infoType)
{
  __cpuid_count(infoType, 0, info[0], info[1], info[2], info[3]);
}

uint64_t _xgetbv(unsigned int index)
{
  uint32_t eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((uint64_t)edx << 32) | eax;
}

#ifndef _XCR_XFEATURE_ENABLED_MASK
#define _XCR_XFEATURE_ENABLED_MASK  0
#endif
#endif

namespace mCpuExtensions
{
  bool sseSupported = false;
  bool sse2Supported = false;
  bool sse3Supported = false;
  bool ssse3Supported = false;
  bool sse41Supported = false;
  bool sse42Supported = false;
  bool avxSupported = false;
  bool avx2Supported = false;
  bool fma3Supported = false;
  bool avx512FSupported = false;
  bool avx512PFSupported = false;
  bool avx512ERSupported = false;
  bool avx512CDSupported = false;
  bool avx512BWSupported = false;
  bool avx512DQSupported = false;
  bool avx512VLSupported = false;
  bool avx512IFMASupported = false;
  bool avx512VBMISupported = false;
  bool avx512VNNISupported = false;
  bool avx512VBMI2Supported = false;
  bool avx512POPCNTDQSupported = false;
  bool avx512BITALGSupported = false;
  bool avx5124VNNIWSupported = false;
  bool avx5124FMAPSSupported = false;
  bool aesNiSupported = false;

  static bool simdFeaturesDetected = false;

  void Detect()
  {
    if (simdFeaturesDetected)
      return;

    int32_t info[4];
    cpuid(info, 0);
    const int32_t idCount = info[0];

    if (idCount >= 0x1)
    {
      int32_t cpuInfo[4];
      cpuid(cpuInfo, 1);

      const bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
      const bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

      if (osUsesXSAVE_XRSTORE && cpuAVXSuport)
      {
        const uint64_t xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
      }

      sseSupported = (cpuInfo[3] & (1 << 25)) != 0;
      sse2Supported = (cpuInfo[3] & (1 << 26)) != 0;
      sse3Supported = (cpuInfo[2] & (1 << 0)) != 0;

      ssse3Supported = (cpuInfo[2] & (1 << 9)) != 0;
      sse41Supported = (cpuInfo[2] & (1 << 19)) != 0;
      sse42Supported = (cpuInfo[2] & (1 << 20)) != 0;
      fma3Supported = (cpuInfo[2] & (1 << 12)) != 0;
      aesNiSupported = (cpuInfo[2] & (1 << 25)) != 0;
    }

    if (idCount >= 0x7)
    {
      int32_t cpuInfo[4];
      cpuid(cpuInfo, 7);

      avx2Supported = (cpuInfo[1] & (1 << 5)) != 0;
      avx512FSupported = (cpuInfo[1] & 1 << 16) != 0;
      avx512PFSupported = (cpuInfo[1] & 1 << 26) != 0;
      avx512ERSupported = (cpuInfo[1] & 1 << 27) != 0;
      avx512CDSupported = (cpuInfo[1] & 1 << 28) != 0;
      avx512BWSupported = (cpuInfo[1] & 1 << 30) != 0;
      avx512DQSupported = (cpuInfo[1] & 1 << 17) != 0;
      avx512VLSupported = (cpuInfo[1] & 1 << 31) != 0;
      avx512IFMASupported = (cpuInfo[1] & 1 << 21) != 0;
      avx512VBMISupported = (cpuInfo[2] & 1 << 1) != 0;
      avx512VNNISupported = (cpuInfo[2] & 1 << 11) != 0;
      avx512VBMI2Supported = (cpuInfo[2] & 1 << 6) != 0;
      avx512POPCNTDQSupported = (cpuInfo[2] & 1 << 14) != 0;
      avx512BITALGSupported = (cpuInfo[2] & 1 << 12) != 0;
      avx5124VNNIWSupported = (cpuInfo[3] & 1 << 2) != 0;
      avx5124FMAPSSupported = (cpuInfo[3] & 1 << 3) != 0;
    }

    simdFeaturesDetected = true;
  }
};

//////////////////////////////////////////////////////////////////////////

int64_t mParseInt(IN const char *start, OPTIONAL OUT const char **pEnd /* = nullptr */)
{
  const char *endIfNoEnd = nullptr;

  if (pEnd == nullptr)
    pEnd = &endIfNoEnd;

  int64_t ret = 0;

  // See: https://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
  int64_t negate = 0;

  if (*start == '-')
  {
    negate = 1;
    start++;
  }

  while (true)
  {
    uint8_t digit = *start - '0';

    if (digit > 9)
      break;

    ret = (ret << 1) + (ret << 3) + digit;
    start++;
  }

  *pEnd = start;

  return (ret ^ -negate) + negate;
}

uint64_t mParseUInt(IN const char *start, OPTIONAL OUT const char **pEnd /* = nullptr */)
{
  const char *endIfNoEnd = nullptr;

  if (pEnd == nullptr)
    pEnd = &endIfNoEnd;

  uint64_t ret = 0;

  while (true)
  {
    uint8_t digit = *start - '0';

    if (digit > 9)
      break;

    ret = (ret << 1) + (ret << 3) + digit;
    start++;
  }

  *pEnd = start;

  return ret;
}

double_t mParseFloat(IN const char *start, OPTIONAL OUT const char **pEnd /* = nullptr */)
{
  const char *endIfNoEnd = nullptr;

  if (pEnd == nullptr)
    pEnd = &endIfNoEnd;

  uint64_t sign = 0;

  if (*start == '-')
  {
    sign = (uint64_t)1 << 63; // IEEE floating point signed bit.
    ++start;
  }

  const char *_end = start;
  const int64_t left = mParseInt(start, &_end);
  double_t ret = (double_t)left;

  if (*_end == '.')
  {
    start = _end + 1;
    const int64_t right = mParseInt(start, &_end);

    const double_t fracMult[] = { 0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13 };

    if (_end - start < (ptrdiff_t)mARRAYSIZE(fracMult))
      ret = (ret + right * fracMult[_end - start]);
    else
      ret = (ret + right * mPow(10, _end - start));

    // Get Sign. (memcpy should get optimized away and is only there to prevent undefined behavior)
    {
      uint64_t data;
      static_assert(sizeof(data) == sizeof(ret), "Platform not supported.");
      memcpy(&data, &ret, sizeof(data));
      data ^= sign;
      memcpy(&ret, &data, sizeof(data));
    }

    *pEnd = _end;

    if (*_end == 'e' || *_end == 'E')
    {
      start = ++_end;

      if ((*start >= '0' && *start <= '9') || *start == '-')
      {
        ret *= mPow(10, mParseInt(start, &_end));

        *pEnd = _end;
      }
    }
  }
  else
  {
    // Get Sign. (memcpy should get optimized away and is only there to prevent undefined behavior)
    {
      uint64_t data;
      static_assert(sizeof(data) == sizeof(ret), "Platform not supported.");
      memcpy(&data, &ret, sizeof(data));
      data ^= sign;
      memcpy(&ret, &data, sizeof(data));
    }

    if (*_end == 'e' || *_end == 'E')
    {
      start = ++_end;

      if ((*start >= '0' && *start <= '9') || *start == '-')
        ret *= mPow(10, mParseInt(start, &_end));
    }

    *pEnd = _end;
  }

  return ret;
}

//////////////////////////////////////////////////////////////////////////

bool mIsInt(IN const char *text)
{
  if (text == nullptr)
    return false;

  return mIsInt(text, strlen(text));
}

bool mIsInt(IN const char *text, const size_t length)
{
  if (text == nullptr)
    return false;

  const bool sign = (text[0] == '-');
  size_t i = (size_t)sign;

  for (; i < length; i++)
  {
    if (text[i] < '0' || text[i] > '9')
    {
      if (text[i] == '\0')
        return (i > (size_t)sign);

      return false;
    }
  }

  return i > (size_t)sign;
}

//////////////////////////////////////////////////////////////////////////

bool mIsUInt(IN const char *text)
{
  if (text == nullptr)
    return false;

  return mIsUInt(text, strlen(text));
}

bool mIsUInt(IN const char *text, const size_t length)
{
  if (text == nullptr)
    return false;
  
  size_t i = 0;

  for (; i < length; i++)
  {
    if (text[i] < '0' || text[i] > '9')
    {
      if (text[i] == '\0')
        return i > 0;

      return false;
    }
  }

  return i > 0;
}

//////////////////////////////////////////////////////////////////////////

bool mIsFloat(IN const char *text)
{
  if (text == nullptr)
    return false;

  return mIsFloat(text, strlen(text));
}

bool mIsFloat(IN const char *text, const size_t length)
{
  if (text == nullptr)
    return false;

  bool hasDigits = false;
  bool hasPostPeriodDigits = false;
  bool hasPostExponentDigits = false;

  size_t i = (size_t)(text[0] == '-');

  for (; i < length; i++)
  {
    if (text[i] == '\0')
    {
      return hasDigits;
    }
    else if (text[i] == '.')
    {
      i++;
      goto period;
    }
    else if (text[i] == 'e' || text[i] == 'E')
    {
      if (!hasDigits)
        return false;
      
      i++;
      goto exponent;
    }
    else if (text[i] < '0' || text[i] > '9')
    {
      return false;
    }
    else
    {
      hasDigits = true;
    }
  }

  return hasDigits;

period:
  for (; i < length; i++)
  {
    if (text[i] == '\0')
    {
      return hasDigits || hasPostPeriodDigits;
    }
    else if (text[i] == 'e' || text[i] == 'E')
    {
      if (!(hasDigits || hasPostPeriodDigits))
        return false;

      i++;
      goto exponent;
    }
    else if (text[i] < '0' || text[i] > '9')
    {
      return false;
    }
    else
    {
      hasPostPeriodDigits = true;
    }
  }

  return hasDigits || hasPostPeriodDigits;

exponent:
  i += (size_t)(text[i] == '-');

  for (; i < length; i++)
  {
    if (text[i] < '0' || text[i] > '9')
    {
      if (text[i] == '\0')
        return hasPostExponentDigits && (hasPostPeriodDigits || hasDigits);

      return false;
    }
    else
    {
      hasPostExponentDigits = true;
    }
  }

  return hasPostExponentDigits && (hasPostPeriodDigits || hasDigits);
}

//////////////////////////////////////////////////////////////////////////

bool mStartsWithInt(IN const char *text)
{
  if (text == nullptr)
    return false;

  return mStartsWithInt(text, strlen(text));
}

bool mStartsWithInt(IN const char *text, const size_t length)
{
  if (text == nullptr)
    return false;

  const bool sign = (text[0] == '-');
  size_t i = (size_t)sign;

  for (; i < length; i++)
    if (text[i] < '0' || text[i] > '9')
      return (i > (size_t)sign);

  return i > (size_t)sign;
}

//////////////////////////////////////////////////////////////////////////

bool mStartsWithUInt(IN const char *text)
{
  if (text == nullptr)
    return false;

  return mStartsWithUInt(text, strlen(text));
}

bool mStartsWithUInt(IN const char *text, const size_t length)
{
  if (text == nullptr)
    return false;

  size_t i = 0;

  for (; i < length; i++)
    if (text[i] < '0' || text[i] > '9')
      return i > 0;

  return i > 0;
}

//////////////////////////////////////////////////////////////////////////

int64_t mParseInt(IN const wchar_t *start, OPTIONAL OUT const wchar_t **pEnd /* = nullptr */)
{
  const wchar_t *endIfNoEnd = nullptr;

  if (pEnd == nullptr)
    pEnd = &endIfNoEnd;

  int64_t ret = 0;

  // See: https://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
  int64_t negate = 0;

  if (*start == L'-')
  {
    negate = 1;
    start++;
  }

  while (true)
  {
    uint16_t digit = *start - L'0';

    if (digit > 9)
      break;

    ret = (ret << 1) + (ret << 3) + digit;
    start++;
  }

  *pEnd = start;

  return (ret ^ -negate) + negate;
}

uint64_t mParseUInt(IN const wchar_t *start, OPTIONAL OUT const wchar_t **pEnd /* = nullptr */)
{
  const wchar_t *endIfNoEnd = nullptr;

  if (pEnd == nullptr)
    pEnd = &endIfNoEnd;

  uint64_t ret = 0;

  while (true)
  {
    uint16_t digit = *start - L'0';

    if (digit > 9)
      break;

    ret = (ret << 1) + (ret << 3) + digit;
    start++;
  }

  *pEnd = start;

  return ret;
}

double_t mParseFloat(IN const wchar_t *start, OPTIONAL OUT const wchar_t **pEnd /* = nullptr */)
{
  const wchar_t *endIfNoEnd = nullptr;

  if (pEnd == nullptr)
    pEnd = &endIfNoEnd;

  uint64_t sign = 0;

  if (*start == L'-')
  {
    sign = (uint64_t)1 << 63; // IEEE floating point signed bit.
    ++start;
  }

  const wchar_t *_end = start;
  const int64_t left = mParseInt(start, &_end);
  double_t ret = (double_t)left;

  if (*_end == L'.')
  {
    start = _end + 1;
    const int64_t right = mParseInt(start, &_end);

    const double_t fracMult[] = { 0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13 };

    if (_end - start < (ptrdiff_t)mARRAYSIZE(fracMult))
      ret = (ret + right * fracMult[_end - start]);
    else
      ret = (ret + right * mPow(10, _end - start));

    // Get Sign. (memcpy should get optimized away and is only there to prevent undefined behavior)
    {
      uint64_t data;
      static_assert(sizeof(data) == sizeof(ret), "Platform not supported.");
      memcpy(&data, &ret, sizeof(data));
      data ^= sign;
      memcpy(&ret, &data, sizeof(data));
    }

    *pEnd = _end;

    if (*_end == L'e' || *_end == L'E')
    {
      start = ++_end;

      if ((*start >= L'0' && *start <= L'9') || *start == L'-')
      {
        ret *= mPow(10, mParseInt(start, &_end));

        *pEnd = _end;
      }
    }
  }
  else
  {
    // Get Sign. (memcpy should get optimized away and is only there to prevent undefined behavior)
    {
      uint64_t data;
      static_assert(sizeof(data) == sizeof(ret), "Platform not supported.");
      memcpy(&data, &ret, sizeof(data));
      data ^= sign;
      memcpy(&ret, &data, sizeof(data));
    }

    if (*_end == L'e' || *_end == L'E')
    {
      start = ++_end;

      if ((*start >= L'0' && *start <= L'9') || *start == L'-')
        ret *= mPow(10, mParseInt(start, &_end));
    }

    *pEnd = _end;
  }

  return ret;
}

//////////////////////////////////////////////////////////////////////////

bool mIsInt(IN const wchar_t *text)
{
  if (text == nullptr)
    return false;

  return mIsInt(text, wcslen(text));
}

bool mIsInt(IN const wchar_t *text, const size_t length)
{
  if (text == nullptr)
    return false;

  const bool sign = (text[0] == L'-');
  size_t i = (size_t)sign;

  for (; i < length; i++)
  {
    if (text[i] < L'0' || text[i] > L'9')
    {
      if (text[i] == L'\0')
        return (i > (size_t)sign);

      return false;
    }
  }

  return i > (size_t)sign;
}

//////////////////////////////////////////////////////////////////////////

bool mIsUInt(IN const wchar_t *text)
{
  if (text == nullptr)
    return false;

  return mIsUInt(text, wcslen(text));
}

bool mIsUInt(IN const wchar_t *text, const size_t length)
{
  if (text == nullptr)
    return false;

  size_t i = 0;

  for (; i < length; i++)
  {
    if (text[i] < L'0' || text[i] > L'9')
    {
      if (text[i] == L'\0')
        return i > 0;

      return false;
    }
  }

  return i > 0;
}

//////////////////////////////////////////////////////////////////////////

bool mIsFloat(IN const wchar_t *text)
{
  if (text == nullptr)
    return false;

  return mIsFloat(text, wcslen(text));
}

bool mIsFloat(IN const wchar_t *text, const size_t length)
{
  if (text == nullptr)
    return false;

  bool hasDigits = false;
  bool hasPostPeriodDigits = false;
  bool hasPostExponentDigits = false;

  size_t i = (size_t)(text[0] == L'-');

  for (; i < length; i++)
  {
    if (text[i] == L'\0')
    {
      return hasDigits;
    }
    else if (text[i] == L'.')
    {
      i++;
      goto period;
    }
    else if (text[i] == L'e' || text[i] == L'E')
    {
      if (!hasDigits)
        return false;

      i++;
      goto exponent;
    }
    else if (text[i] < L'0' || text[i] > L'9')
    {
      return false;
    }
    else
    {
      hasDigits = true;
    }
  }

  return hasDigits;

period:
  for (; i < length; i++)
  {
    if (text[i] == L'\0')
    {
      return hasDigits || hasPostPeriodDigits;
    }
    else if (text[i] == L'e' || text[i] == L'E')
    {
      if (!(hasDigits || hasPostPeriodDigits))
        return false;

      i++;
      goto exponent;
    }
    else if (text[i] < L'0' || text[i] > L'9')
    {
      return false;
    }
    else
    {
      hasPostPeriodDigits = true;
    }
  }

  return hasDigits || hasPostPeriodDigits;

exponent:
  i += (size_t)(text[i] == L'-');

  for (; i < length; i++)
  {
    if (text[i] < L'0' || text[i] > L'9')
    {
      if (text[i] == L'\0')
        return hasPostExponentDigits && (hasPostPeriodDigits || hasDigits);

      return false;
    }
    else
    {
      hasPostExponentDigits = true;
    }
  }

  return hasPostExponentDigits && (hasPostPeriodDigits || hasDigits);
}

//////////////////////////////////////////////////////////////////////////

bool mStartsWithInt(IN const wchar_t *text)
{
  if (text == nullptr)
    return false;

  return mStartsWithInt(text, wcslen(text));
}

bool mStartsWithInt(IN const wchar_t *text, const size_t length)
{
  if (text == nullptr)
    return false;

  const bool sign = (text[0] == L'-');
  size_t i = (size_t)sign;

  for (; i < length; i++)
    if (text[i] < L'0' || text[i] > L'9')
      return (i > (size_t)sign);

  return i > (size_t)sign;
}

//////////////////////////////////////////////////////////////////////////

bool mStartsWithUInt(IN const wchar_t *text)
{
  if (text == nullptr)
    return false;

  return mStartsWithUInt(text, wcslen(text));
}

bool mStartsWithUInt(IN const wchar_t *text, const size_t length)
{
  if (text == nullptr)
    return false;

  size_t i = 0;

  for (; i < length; i++)
    if (text[i] < L'0' || text[i] > L'9')
      return i > 0;

  return i > 0;
}

//////////////////////////////////////////////////////////////////////////

uint64_t mRnd()
{
  mALIGN(16) static uint64_t last[2] = { (uint64_t)mGetCurrentTimeNs(), __rdtsc() };

  if (!mCpuExtensions::simdFeaturesDetected) // let's hope that's faster than the function call...
    mCpuExtensions::Detect();
  
  if (mCpuExtensions::aesNiSupported)
  {
    // If we can use AES-NI, use aes decryption.
  
    mALIGN(16) static uint64_t last2[2] = { ~__rdtsc(), ~(uint64_t)mGetCurrentTimeMs() };
    
    const __m128i a = _mm_load_si128(reinterpret_cast<__m128i *>(last));
    const __m128i b = _mm_load_si128(reinterpret_cast<__m128i *>(last2));
    
    const __m128i r = _mm_aesdec_si128(a, b);
    
    _mm_store_si128(reinterpret_cast<__m128i *>(last), b);
    _mm_store_si128(reinterpret_cast<__m128i *>(last2), r);
  
    return last[1] ^ last[0];
  }
  else
  {
    // This is simply PCG, which is about 25% slower, (we're talking ~5 ns/call, so it's certainly not slow) but works on all CPUs as a fallback option.

    const uint64_t oldstate_hi = last[0];
    const uint64_t oldstate_lo = oldstate_hi * 6364136223846793005ULL + (last[1] | 1);
    last[0] = oldstate_hi * 6364136223846793005ULL + (last[1] | 1);
    
    const uint32_t xorshifted_hi = (uint32_t)(((oldstate_hi >> 18) ^ oldstate_hi) >> 27);
    const uint32_t rot_hi = (uint32_t)(oldstate_hi >> 59);
    
    const uint32_t xorshifted_lo = (uint32_t)(((oldstate_lo >> 18) ^ oldstate_lo) >> 27);
    const uint32_t rot_lo = (uint32_t)(oldstate_lo >> 59);
    
    const uint32_t hi = (xorshifted_hi >> rot_hi) | (xorshifted_hi << (uint32_t)((-(int32_t)rot_hi) & 31));
    const uint32_t lo = (xorshifted_lo >> rot_lo) | (xorshifted_lo << (uint32_t)((-(int32_t)rot_lo) & 31));
  
    return ((uint64_t)hi << 32) | lo;
  }
}
