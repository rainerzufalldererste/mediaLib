#include <windows.h>
#include <winnt.h>
#include <intsafe.h>
#include <cstdio>
#include <chrono>

#include "mediaLib.h"

#ifdef mPLATFORM_WINDOWS
HANDLE mStdOutHandle = nullptr;
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

  printf("\x1b[%" PRIu64 ";%" PRIu64 "m", fgColour < 0x8 ? (30 + fgColour) : (90 - 8 + fgColour) | bgColour < 0x8 ? (30 + bgColour) : (90 - 8 + bgColour));
#endif
}

void mDefaultPrint(const char *text)
{
  fputs(text, stdout);
}

void mDebugPrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  fputs("[Debug] ", stdout);
  mSetConsoleColour(mCC_DarkGray, mCC_Black);
  fputs(text, stdout);
  mResetConsoleColour();

  mDebugOut("[Debug] %s", text);
}

void mLogPrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  fputs("[Log]   ", stdout);
  mSetConsoleColour(mCC_DarkCyan, mCC_Black);
  fputs(text, stdout);
  mResetConsoleColour();

  mDebugOut("[Log]   %s", text);
}

void mErrorPrint(const char *text)
{
  mSetConsoleColour(mCC_BrightRed, mCC_Black);
  fputs("[Error] ", stdout);
  mSetConsoleColour(mCC_BrightYellow, mCC_DarkRed);
  fputs(text, stdout);
  mResetConsoleColour();

  mDebugOut("[Error] %s", text);
}

void mTracePrint(const char *text)
{
  mSetConsoleColour(mCC_White, mCC_Black);
  fputs("[Trace] ", stdout);
  mSetConsoleColour(mCC_BrightGray, mCC_DarkGray);
  fputs(text, stdout);
  mResetConsoleColour();

  mDebugOut("[Trace] %s", text);
}

mPrintCallbackFunc *mPrintCallback = &mDefaultPrint;
mPrintCallbackFunc *mPrintErrorCallback = &mErrorPrint;
mPrintCallbackFunc *mPrintLogCallback = &mLogPrint;
mPrintCallbackFunc *mPrintDebugCallback = &mDebugPrint;
mPrintCallbackFunc *mPrintTraceCallback = &mTracePrint;

void mPrintPrepare(mPrintCallbackFunc *pFunc, const char *format, ...)
{
  if (pFunc != nullptr)
  {
    char buffer[1024 * 16];

    va_list args;
    va_start(args, format);
    vsprintf_s(buffer, format, args);
    va_end(args);

    buffer[mARRAYSIZE(buffer) - 1] = '\0';
    
    (*pFunc)(buffer);
  }
}

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
    }

    simdFeaturesDetected = true;
  }
};

int64_t mParseInt(IN const char *start, OUT const char **pEnd)
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

double_t mParseFloat(IN const char *start, OUT const char **pEnd)
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

    if (_end - start < mARRAYSIZE(fracMult))
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
