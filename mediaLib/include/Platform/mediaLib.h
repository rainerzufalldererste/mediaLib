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

#ifdef _MSC_VER
#include <windows.h>
#endif // !_MSC_VER

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

#ifdef _MSC_VER
#define mDLL_FUNC_PREFIX __dll__
#define mDLL_TYPEDEF_POSTFIX __dll__FUNC_TYPE__

#define mTYPEDEF_DLL_FUNC(name, returnType, ...) typedef returnType mCONCAT_LITERALS(name, mDLL_TYPEDEF_POSTFIX) (__VA_ARGS__)
#define mLOAD_FROM_DLL(symbol, module) do { mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) = (mCONCAT_LITERALS(symbol, mDLL_TYPEDEF_POSTFIX) *)GetProcAddress(module, #symbol); if(mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) == nullptr) { DWORD errorCode = GetLastError(); mASSERT(mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) != nullptr, "Symbol '" #symbol "' could not be loaded from dynamic library. (errorcode: 0x%" PRIx64 ")", (uint64_t)errorCode); } } while (0)
#define mDLL_DEFINE_SYMBOL(symbol) mCONCAT_LITERALS(symbol, mDLL_TYPEDEF_POSTFIX) *mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) = nullptr
#define mDLL_CALL(symbol, ...) ((*mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol))(__VA_ARGS__))
#endif

#endif // !_DEPENDENCIES_DEFINED

#ifndef default_h__
#define default_h__

#include "mTypeTraits.h"
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
