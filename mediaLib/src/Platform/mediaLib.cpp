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
