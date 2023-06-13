#include <windows.h>
#include <intsafe.h>
#include <cstdio>
#include "mediaLib.h"

void mDefaultPrint(const char *text)
{
  printf(text);
}

mPrintCallbackFunc *mPrintCallback = &mDefaultPrint;
mPrintCallbackFunc *mPrintErrorCallback = &mDefaultPrint;
mPrintCallbackFunc *mPrintLogCallback = &mDefaultPrint;
mPrintCallbackFunc *mPrintDebugCallback = &mDefaultPrint;
mPrintCallbackFunc *mPrintTraceCallback = &mDefaultPrint;

void mPrintPrepare(mPrintCallbackFunc *pFunc, const char *format, ...)
{
  if (pFunc != nullptr)
  {
    char buffer[1024 * 8];

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
