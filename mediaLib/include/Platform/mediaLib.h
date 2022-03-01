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

#ifndef CONST_FIELD
 #define CONST_FIELD
#endif // !CONST_FIELD

#ifndef PUBLIC_FIELD
 #define PUBLIC_FIELD
#endif // !PUBLIC_FIELD

#define mVECTORCALL __vectorcall
 #define mINLINE __forceinline
#define mALIGN(x) __declspec(align(x))

#if defined(_MSC_VER) && _MSC_VER < 1920
 #define noexcept
#endif

#if defined(_MSVC_LANG)
 #if _MSVC_LANG < 201703L
  #define mIF_CONSTEXPR if
 #else
  #define mIF_CONSTEXPR if constexpr
 #endif
#else
 #if defined(__cplusplus)
  #if __cplusplus < 201703L
   #define mIF_CONSTEXPR if
  #else
   #define mIF_CONSTEXPR if constexpr
  #endif
 #endif
#endif

#define mARRAYSIZE_C_STYLE(arrayName) (sizeof(arrayName) / sizeof(arrayName[0]))

#ifdef mFORCE_ARRAYSIZE_C_STYLE
#define mARRAYSIZE(arrayName) mARRAYSIZE_C_STYLE(arrayName)
#else
template <typename T, size_t TCount>
inline constexpr size_t mARRAYSIZE(const T(&)[TCount]) { return TCount; }
#endif

#define mBITS_OF(type) (sizeof(type) * CHAR_BIT)

#ifndef __M_FILE__
#define __M_FILE__ __FILE__
#endif

#define mCONCAT_LITERALS_INTERNAL(x, y) x ## y
#define mCONCAT_LITERALS(x, y) mCONCAT_LITERALS_INTERNAL(x, y)
#define mCONCAT_LITERALS_INTERNAL3(x, y, z) x ## y ## z
#define mCONCAT_LITERALS3(x, y, z) mCONCAT_LITERALS_INTERNAL3(x, y, z)
#define mSTRINGIFY(x) #x
#define mSTRINGIFY_VALUE(x) mSTRINGIFY(x)

#ifndef GIT_BUILD

#define mASSERT(expression, text) \
  do \
  { if(!(expression)) \
    { const char *__text__ = (text); \
      mPRINT_ERROR("Assertion Failed: " #expression ""); \
      mPRINT_ERROR(__text__); \
      mPRINT_ERROR("In File '" __M_FILE__ "' : Line '" mSTRINGIFY_VALUE(__LINE__) "' (Function '" __FUNCTION__ "')"); \
      mAssert_Internal(#expression, __text__, __FUNCTION__, __M_FILE__, __LINE__); \
    } \
  } while (0)

#define mFAIL(text) \
  do \
  { const char *__text__ = (text); \
    mPRINT_ERROR("Assertion Failed:"); \
    mPRINT_ERROR(__text__); \
    mPRINT_ERROR("In File '" __M_FILE__ "' : Line '" mSTRINGIFY_VALUE(__LINE__) "' (Function '" __FUNCTION__ "')"); \
    mFail_Internal(__text__, __FUNCTION__, __M_FILE__, __LINE__); \
  } while (0)

#else

#define mASSERT(expression, text) \
  do \
  { if(!(expression)) \
    { const char *__text__ = (text); \
      mPRINT_ERROR("Assertion Failed."); \
      mPRINT_ERROR(__text__); \
      mPRINT_ERROR("In File '" __M_FILE__ "' : Line '" mSTRINGIFY_VALUE(__LINE__) "'"); \
      mAssert_Internal("<EXPRESSION_NOT_AVAILABLE>", __text__, "<FUNCTION_NAME_NOT_AVAILABLE>", __M_FILE__, __LINE__); \
    } \
  } while (0)

#define mFAIL(text, ...) \
  do \
  { const char *__text__ = (text); \
    mPRINT_ERROR("Assertion Failed."); \
    mPRINT_ERROR(__text__); \
    mPRINT_ERROR("In File '" __M_FILE__ "' : Line '" mSTRINGIFY_VALUE(__LINE__) "'"); \
    mFail_Internal(__text__, "<FUNCTION_NAME_NOT_AVAILABLE>", __M_FILE__, __LINE__); \
  } while (0)

#endif

#define mPRINT(text, ...) mPrintToFunction(mPrintCallback, text, __VA_ARGS__)
#define mLOG(text, ...) mPrintToFunction(mPrintLogCallback, text, __VA_ARGS__)
#define mPRINT_ERROR(text, ...) mPrintToFunction(mPrintErrorCallback, text, __VA_ARGS__)

//#define mTRACE_ENABLED

#if defined (_DEBUG)
 #if !defined(mTRACE_ENABLED)
  #define mTRACE_ENABLED
 #endif

 #define mPRINT_DEBUG(text, ...) mPrintToFunction(mPrintDebugCallback, text, __VA_ARGS__)
#else
 #define mPRINT_DEBUG(text, ...) mUnused(text, __VA_ARGS__)
#endif

#if defined (mTRACE_ENABLED)
 #define mTRACE(text, ...) mPrintToFunction(mPrintTraceCallback, text, __VA_ARGS__)
#else
 #define mTRACE(text, ...) mUnused(text, __VA_ARGS__)
#endif

#define mSTATIC_ASSERT(expression, text) static_assert(expression, __FUNCSIG__ " : " text)

template <typename T>
void mUnused(const T &unused)
{
  (void)unused;
}

template <typename T, typename ...Args>
void mUnused(const T &unused, Args && ...args)
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
void mSilencablePrint(const char *text);
void mSilencablePrintError(const char *text);
void mSilencablePrintLog(const char *text);
void mSilencablePrintDebug(const char *text);
void mSilencablePrintTrace(const char *text);

inline void mPrintToFunction(mPrintCallbackFunc *pFunc, const char *text)
{
  if (pFunc != nullptr)
    (*pFunc)(text);
}

template <typename ...Args>
inline void mPrintToFunction(mPrintCallbackFunc *pFunc, Args && ... args)
{
  if (pFunc != nullptr)
    (*pFunc)(mFormat(args...));
}

#ifdef mPLATFORM_WINDOWS
void mPrintToOutputWithLength(const char *text, const size_t length);
#endif
void mPrintToOutput(const char *text);

template <size_t Tcount>
inline void mPrintToOutputArray(const char(&text)[Tcount])
{
  if (text == nullptr)
    return;

#ifdef mPLATFORM_WINDOWS
  mPrintToOutputWithLength(text, strnlen(text, Tcount));
#else
  mPrintToOutput(text);
#endif
}

enum mResult mSetOutputFilePath(const struct mString &path, const bool append = true);
void mResetOutputFile();
void mFlushOutput();

#ifdef _DEBUG
 #define mASSERT_DEBUG(expr, text) mASSERT(expr, text)
 #define mFAIL_DEBUG(text) mFAIL(text)
#else // !_DEBUG
 #define mASSERT_DEBUG(expr, text)
 #define mFAIL_DEBUG(text)
#endif

#ifdef mPLATFORM_WINDOWS
 #define mDLL_FUNC_PREFIX __dll__

 #define mLOAD_FROM_DLL(symbol, module) \
   do \
   { mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) = (decltype(symbol) *)GetProcAddress(module, #symbol); \
     if (mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol) == nullptr) \
     { const DWORD errorCode = GetLastError(); \
       mPRINT_ERROR("Symbol '" #symbol "' could not be loaded from dynamic library. (Error Code: 0x", mFUInt<mFHex>(errorCode), ")"); \
       mRETURN_RESULT(mR_ResourceNotFound); \
     } \
   } while (0)

 #define mDLL_NAME(symbol) mCONCAT_LITERALS(mDLL_FUNC_PREFIX, symbol)
 #define mDLL_DEFINE_SYMBOL(symbol) decltype(symbol) *mDLL_NAME(symbol) = nullptr
 #define mDLL_CALL(symbol, ...) ((*mDLL_NAME(symbol))(__VA_ARGS__))

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
#include "mString.h" // will include "mFormat.h"

mFUNCTION(mSleep, const size_t milliseconds = 0);
void mAssert_Internal(const char *expression, const char *text, const char *function, const char *file, const int32_t line);
void mFail_Internal(const char *text, const char *function, const char *file, const int32_t line);
int64_t mGetCurrentTimeMs();
int64_t mGetCurrentTimeNs();

enum mConsoleColour
{
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
  mCC_White
};

void mCreateConsole();
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
  extern bool avx512FSupported;
  extern bool avx512PFSupported;
  extern bool avx512ERSupported;
  extern bool avx512CDSupported;
  extern bool avx512BWSupported;
  extern bool avx512DQSupported;
  extern bool avx512VLSupported;
  extern bool avx512IFMASupported;
  extern bool avx512VBMISupported;
  extern bool avx512VNNISupported;
  extern bool avx512VBMI2Supported;
  extern bool avx512POPCNTDQSupported;
  extern bool avx512BITALGSupported;
  extern bool avx5124VNNIWSupported;
  extern bool avx5124FMAPSSupported;
  extern bool aesNiSupported;

  void Detect();
};

int64_t mParseInt(IN const char *start, OPTIONAL OUT const char **pEnd = nullptr);
uint64_t mParseUInt(IN const char *start, OPTIONAL OUT const char **pEnd = nullptr);
double_t mParseFloat(IN const char *start, OPTIONAL OUT const char **pEnd = nullptr);

bool mIsInt(IN const char *text);
bool mIsInt(IN const char *text, const size_t length);
bool mIsUInt(IN const char *text);
bool mIsUInt(IN const char *text, const size_t length);
bool mIsFloat(IN const char *text);
bool mIsFloat(IN const char *text, const size_t length);

bool mStartsWithInt(IN const char *text);
bool mStartsWithInt(IN const char *text, const size_t length);
bool mStartsWithUInt(IN const char *text);
bool mStartsWithUInt(IN const char *text, const size_t length);

uint64_t mRnd();

#endif // mediaLib_h__
