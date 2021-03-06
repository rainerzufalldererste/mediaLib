// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _HAS_EXCEPTIONS
#undef _HAS_EXCEPTIONS
#endif

#define _HAS_EXCEPTIONS 0

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES 1
#endif // !_USE_MATH_DEFINES

#include <assert.h>
#include <stdint.h>
#include <climits>
#include <math.h>
#include <functional>
#include <type_traits>
#include <inttypes.h>
#include <float.h>

#ifndef _DEPENDENCIES_DEFINED
#define _DEPENDENCIES_DEFINED

#ifndef RETURN
#define RETURN return
#endif // !RETURN

#ifndef IN
#define IN
#endif // !IN

#ifndef OUT
#define OUT
#endif // !OUT

#ifndef IN_OUT
#define IN_OUT IN OUT
#endif // !IN_OUT

#ifndef OPTIONAL
#define OPTIONAL
#endif // !OPTIONAL

#define mVECTORCALL __vectorcall
#define mINLINE __forceinline

#define mARRAYSIZE(arrayName) (sizeof(arrayName) / sizeof(arrayName[0]))
#define mBYTES_OF(type) (sizeof(type) << 3)

#define mASSERT(expression, text, ...) \
  do \
  { if(!(expression)) \
    { char buffer[1024 * 8]; \
      sprintf_s(buffer, text, __VA_ARGS__); \
      buffer[mARRAYSIZE(buffer) - 1] = '\0'; \
      mPRINT_ERROR("Assertion Failed: %s\n'%s'\n\nIn File '%s' : Line '%" PRIi32 "' (Function '%s')\n", #expression, buffer, __FILE__, __LINE__, __FUNCTION__); \
      mAssert_Internal(#expression, buffer, __FUNCTION__, __FILE__, __LINE__); \
    } \
  } while (0)

#define mFAIL(text, ...) \
  do \
  { char buffer[1024 * 8]; \
    sprintf_s(buffer, text, __VA_ARGS__); \
    buffer[mARRAYSIZE(buffer) - 1] = '\0'; \
    mPRINT_ERROR("Assertion Failed: '%s'\n\nIn File '%s' : Line '%" PRIi32 "' (Function '%s')\n", buffer, __FILE__, __LINE__, __FUNCTION__); \
    __debugbreak(); \
  } while (0)

#define mPRINT(text, ...) mPrintPrepare(mPrintCallback, text, __VA_ARGS__)
#define mLOG(text, ...) mPrintPrepare(mPrintLogCallback, text, __VA_ARGS__)
#define mPRINT_ERROR(text, ...) mPrintPrepare(mPrintErrorCallback, text, __VA_ARGS__)

//#define mTRACE_ENABLED

#if defined (_DEBUG)
#if !defined(mTRACE_ENABLED)
#define mTRACE_ENABLED
#endif

#define mPRINT_DEBUG(text, ...) mPrintPrepare(mPrintDebugCallback, text, __VA_ARGS__)
#else
#define mPRINT_DEBUG(text, ...) mUnused(text, __VA_ARGS__)
#endif

#if defined (mTRACE_ENABLED)
#define mTRACE(text, ...) mPrintPrepare(mPrintTraceCallback, text, __VA_ARGS__)
#else
#define mTRACE(text, ...) mUnused(text, __VA_ARGS__)
#endif

template <typename T>
void mUnused(T unused)
{
  (void)unused;
}

template <typename T, typename ...Args>
void mUnused(T unused, Args && ...args)
{
  (void)unused;
  mUnused(args...);
}

typedef void mPrintCallbackFunc(const char *);
extern mPrintCallbackFunc *mPrintCallback;
extern mPrintCallbackFunc *mPrintErrorCallback;
extern mPrintCallbackFunc *mPrintLogCallback;
extern mPrintCallbackFunc *mPrintDebugCallback;
extern mPrintCallbackFunc *mPrintTraceCallback;

void mDefaultPrint(const char *text);
void mPrintPrepare(mPrintCallbackFunc *pFunc, const char *format, ...);

#ifdef _DEBUG
#define mASSERT_DEBUG(expr, text, ...) mASSERT(expr, text, __VA_ARGS__)
#define mFAIL_DEBUG(text, ...) mFAIL(text, __VA_ARGS__)
#else // !_DEBUG
#define mASSERT_DEBUG(expr, text, ...) mUnused(expr, text, __VA_ARGS__)
#define mFAIL_DEBUG(text, ...) mUnused(text, __VA_ARGS__)
#endif

#endif // !_DEPENDENCIES_DEFINED

#ifndef default_h__
#define default_h__

#include "mResult.h"
#include "mDefer.h"
#include "mDestruct.h"
#include "mMemory.h"
#include "mAllocator.h"
#include "mSharedPointer.h"
#include "mMath.h"
#include "mFastMath.h"
#include "mStaticIf.h"
#include "mString.h"

mFUNCTION(mSleep, const size_t milliseconds = 0);
void mAssert_Internal(const char *expression, const char *text, const char *function, const char *file, const int32_t line);

#endif // default_h__
