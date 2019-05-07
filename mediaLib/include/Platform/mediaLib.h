#ifdef _HAS_EXCEPTIONS
#undef _HAS_EXCEPTIONS
#endif

#define _HAS_EXCEPTIONS 0

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES 1
#endif // !_USE_MATH_DEFINES

#ifndef _DEPENDENCIES_DEFINED
#define _DEPENDENCIES_DEFINED

#include <assert.h>
#include <stdint.h>
#include <climits>
#include <math.h>
#include <functional>
#include <type_traits>
#include <inttypes.h>
#include <float.h>
#include <memory.h>
#include <malloc.h>

#ifndef mPLATFORM_WINDOWS
#if defined(_WIN64) || defined(_WIN32)
#define mPLATFORM_WINDOWS 1
#endif
#endif

#ifndef mPLATFORM_UNIX
#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#define mPLATFORM_UNIX 1
#endif
#endif

#ifndef mPLATFORM_APPLE
#if defined(__APPLE__) && defined(__MACH__)
#define mPLATFORM_APPLE 1
#endif
#endif

#ifndef mPLATFORM_LINUX
#if defined(__linux__)
#define mPLATFORM_LINUX 1
#endif
#endif

#ifndef mPLATFORM_BSD
#if defined(BSD)
#define mPLATFORM_BSD 1
#endif
#endif

#ifdef mPLATFORM_WINDOWS
#include <windows.h>
#endif // !mPLATFORM_WINDOWS

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
#define mALIGN(x) __declspec(align(x))

#define mARRAYSIZE(arrayName) (sizeof(arrayName) / sizeof(arrayName[0]))
#define mBYTES_OF(type) (sizeof(type) << 3)

#define mASSERT(expression, text, ...) \
  do \
  { if(!(expression)) \
    { char ___buffer___[1024 * 8]; \
      sprintf_s(___buffer___, text, __VA_ARGS__); \
      ___buffer___[mARRAYSIZE(___buffer___) - 1] = '\0'; \
      mPRINT_ERROR("Assertion Failed: %s\n'%s'\n\nIn File '%s' : Line '%" PRIi32 "' (Function '%s')\n", #expression, ___buffer___, __FILE__, __LINE__, __FUNCTION__); \
      mAssert_Internal(#expression, ___buffer___, __FUNCTION__, __FILE__, __LINE__); \
    } \
  } while (0)

#define mFAIL(text, ...) \
  do \
  { char ___buffer___[1024 * 8]; \
    sprintf_s(___buffer___, text, __VA_ARGS__); \
    ___buffer___[mARRAYSIZE(___buffer___) - 1] = '\0'; \
    mPRINT_ERROR("Assertion Failed: '%s'\n\nIn File '%s' : Line '%" PRIi32 "' (Function '%s')\n", ___buffer___, __FILE__, __LINE__, __FUNCTION__); \
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

#define mSTATIC_ASSERT(expression, text) static_assert(expression, __FUNCSIG__ " : " text)

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

#define mCONCAT_LITERALS_INTERNAL(x, y) x ## y
#define mCONCAT_LITERALS(x, y) mCONCAT_LITERALS_INTERNAL(x, y)
#define mSTRINGIFY(x) #x

#ifdef mPLATFORM_WINDOWS
#define mDLL_FUNC_PREFIX __dll__
#define mDLL_TYPEDEF_POSTFIX __dll__FUNC_TYPE__

#define mTYPEDEF_DLL_FUNC(name, returnType, ...) typedef returnType mCONCAT_LITERALS(name, mDLL_TYPEDEF_POSTFIX) (__VA_ARGS__)

#define mLOAD_FROM_DLL(symbol, module) \
  do \
  { mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) = (mCONCAT_LITERALS(symbol, mDLL_TYPEDEF_POSTFIX) *)GetProcAddress(module, #symbol); \
    if (mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) == nullptr) \
    { const DWORD errorCode = GetLastError(); \
      mPRINT_ERROR("Symbol '" #symbol "' could not be loaded from dynamic library. (error code: 0x%" PRIx64 ")\n", (uint64_t)errorCode); \
      mRETURN_RESULT(mR_ResourceNotFound); \
    } \
  } while (0)

#define mDLL_DEFINE_SYMBOL(symbol) mCONCAT_LITERALS(symbol, mDLL_TYPEDEF_POSTFIX) *mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) = nullptr
#define mDLL_CALL(symbol, ...) ((*mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol))(__VA_ARGS__))

#endif // !mPLATFORM_WINDOWS

#endif // !_DEPENDENCIES_DEFINED

#ifndef mediaLib_h__
#define mediaLib_h__

#include "mTypeTraits.h"
#include "mResult.h"
#include "mDefer.h"
#include "mDestruct.h"
#include "mMemory.h"
#include "mAllocator.h"
#include "mSharedPointer.h"
#include "mMath.h"
#include "mFastMath.h"
#include "mString.h"

mFUNCTION(mSleep, const size_t milliseconds = 0);
void mAssert_Internal(const char *expression, const char *text, const char *function, const char *file, const int32_t line);
int64_t mGetCurrentTimeMs();
int64_t mGetCurrentTimeNs();

enum mConsoleColour
{
#ifdef mPLATFORM_WINDOWS
  mCC_Black,
  mCC_DarkBlue,
  mCC_DarkGreen,
  mCC_DarkCyan,
  mCC_DarkRed,
  mCC_DarkMagenta,
  mCC_DarkYellow,
  mCC_BrightGray,
  mCC_DarkGray,
  mCC_BrightBlue,
  mCC_BrightGreen,
  mCC_BrightCyan,
  mCC_BrightRed,
  mCC_BrightMagenta,
  mCC_BrightYellow,
  mCC_White,
#else
  mCC_Black,
  mCC_DarkRed,
  mCC_DarkGreen,
  mCC_DarkYellow,
  mCC_DarkBlue,
  mCC_DarkMagenta,
  mCC_DarkCyan,
  mCC_BrightGray,
  mCC_DarkGray,
  mCC_BrightRed,
  mCC_BrightGreen,
  mCC_BrightYellow,
  mCC_BrightBlue,
  mCC_BrightMagenta,
  mCC_BrightCyan,
  mCC_White,
#endif
};

void mResetConsoleColour();
void mSetConsoleColour(const mConsoleColour foregroundColour, const mConsoleColour backgroundColour);

namespace mCpuExtensions
{
  extern bool sseSupported;
  extern bool sse2Supported;
  extern bool sse3Supported;
  extern bool ssse3Supported;
  extern bool sse41Supported;
  extern bool sse42Supported;
  extern bool avxSupported;
  extern bool avx2Supported;
  extern bool fma3Supported;

  void Detect();
};

#endif // mediaLib_h__
