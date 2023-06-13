// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <windows.h>
#include <intsafe.h>
#include <cstdio>

#include "default.h"

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
