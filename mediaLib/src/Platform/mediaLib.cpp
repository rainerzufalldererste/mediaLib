#include <windows.h>
#include <winnt.h>
#include <intsafe.h>
#include <cstdio>
#include <chrono>

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "hJxXtXZLn5mrTXicBqpRmqzOi/HGriWHIaPhBoYJdH/uA6vV5/aWehfGSsFZG41NQnPfqNvSkmhJriq2"
#endif

#ifdef mPLATFORM_WINDOWS
HANDLE mStdOutHandle = nullptr;
HANDLE mFileOutHandle = nullptr;
#endif

void mResetConsoleColour()
{
#ifdef mPLATFORM_WINDOWS
  if (mStdOutHandle == nullptr)
    mStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

  if (mStdOutHandle != nullptr && mStdOutHandle != INVALID_HANDLE_VALUE)
    SetConsoleTextAttribute(mStdOutHandle, mCC_BrightGray | (mCC_Black << 8));
#else
  fputs("\x1b[0m", stdout);
#endif
}

void mSetConsoleColour(const mConsoleColour foregroundColour, const mConsoleColour backgroundColour)
{
#ifdef mPLATFORM_WINDOWS
  if (mStdOutHandle == nullptr)
    mStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

  const WORD fgColour = (foregroundColour & 0xF);
  const WORD bgColour = (backgroundColour & 0xF);

  if (mStdOutHandle != nullptr && mStdOutHandle != INVALID_HANDLE_VALUE)
    SetConsoleTextAttribute(mStdOutHandle, fgColour | (bgColour << 4));
#else
  const size_t fgColour = (foregroundColour & 0xF);
  const size_t bgColour = (backgroundColour & 0xF);

  printf("\x1b[%" PRIu64 ";%" PRIu64 "m", fgColour < 0x8 ? (30 + fgColour) : (90 - 8 + fgColour), bgColour < 0x8 ? (30 + bgColour) : (90 - 8 + bgColour));
#endif
}

#ifdef mPLATFORM_WINDOWS
void mPrintToOutputWithLength(const char *text, const size_t length)
{
  if (mStdOutHandle == nullptr)
    mStdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

  if (mStdOutHandle != nullptr && mStdOutHandle != INVALID_HANDLE_VALUE)
    WriteConsoleA(mStdOutHandle, text, (DWORD)length, nullptr, nullptr);

  if (mFileOutHandle != nullptr && mFileOutHandle != INVALID_HANDLE_VALUE)
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

#ifndef GIT_BUILD
  mDebugOut("[Debug] ");
  mDebugOut(text);
#endif
}

void mLogPrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  mPrintToOutputArray("[Log]   ");
  mSetConsoleColour(mCC_DarkCyan, mCC_Black);
  mPrintToOutput(text);
  mResetConsoleColour();

#ifndef GIT_BUILD
  mDebugOut("[Log]   ");
  mDebugOut(text);
#endif
}

void mErrorPrint(const char *text)
{
  mSetConsoleColour(mCC_BrightRed, mCC_Black);
  mPrintToOutputArray("[Error] ");
  mSetConsoleColour(mCC_BrightYellow, mCC_DarkRed);
  mPrintToOutput(text);
  mResetConsoleColour();

#ifndef GIT_BUILD
  mDebugOut("[Error] ");
  mDebugOut(text);
#endif
}

void mTracePrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  mPrintToOutputArray("[Trace] ");
  mSetConsoleColour(mCC_BrightGray, mCC_DarkGray);
  mPrintToOutput(text);
  mResetConsoleColour();

#ifndef GIT_BUILD
  mDebugOut("[Trace] ");
  mDebugOut(text);
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

mFUNCTION(mSleep, const size_t milliseconds)
{
  mFUNCTION_SETUP();

  mERROR_IF(milliseconds > DWORD_MAX, mR_ArgumentOutOfBounds);

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

  void Detect()
  {
    static bool simdFeaturesDetected = false;

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
