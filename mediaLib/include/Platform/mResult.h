#ifndef mResult_h__
#define mResult_h__

#include <functional>
#include <type_traits>
#include <windows.h>
#include <debugapi.h>
#include <stdarg.h>

enum mResult
{
  mR_Success,

  mR_Break,

  mR_InvalidParameter,
  mR_ArgumentNull,
  mR_InternalError,
  mR_MemoryAllocationFailure,
  mR_NotImplemented,
  mR_NotInitialized,
  mR_IndexOutOfBounds,
  mR_ArgumentOutOfBounds,
  mR_Timeout,
  mR_OperationNotSupported,
  mR_ResourceNotFound,
  mR_ResourceInvalid,
  mR_ResourceStateInvalid,
  mR_ResourceIncompatible,
  mR_EndOfStream,
  mR_RenderingError,
  mR_Failure,
  mR_NotSupported,
  mR_ResourceAlreadyExists,
  mR_IOFailure,

  mResult_Count
};

#define mSTDRESULT __result
#define mSUCCEEDED(result) ((result) == mR_Success)
#define mFAILED(result) (!mSUCCEEDED(result))

#define mFUNCTION(FunctionName, ...) mResult FunctionName(__VA_ARGS__)
#define mFUNCTION_SETUP() mResult mSTDRESULT = mR_Success; mUnused(mSTDRESULT)
#define mRESULT_RETURN_RESULT_RAW(result) do { mSTDRESULT = result; return mSTDRESULT; } while (0)
#define mRETURN_SUCCESS() mRESULT_RETURN_RESULT_RAW(mR_Success)

extern const char *g_mResult_lastErrorFile;
extern size_t g_mResult_lastErrorLine;
extern mResult g_mResult_lastErrorResult;

extern bool g_mResult_breakOnError;

#ifdef _DEBUG
#define mBREAK_ON_FAILURE true
#else
#define mBREAK_ON_FAILURE false
#endif // _DEBUG

#ifdef mBREAK_ON_FAILURE
#ifdef mPLATFORM_WINDOWS
#define mDEBUG_BREAK() \
      do \
      { BOOL __isRemoteDebuggerPresent__ = false; \
        if (!CheckRemoteDebuggerPresent(GetCurrentProcess(), &__isRemoteDebuggerPresent__)) \
          __isRemoteDebuggerPresent__ = false; \
        if (IsDebuggerPresent() || __isRemoteDebuggerPresent__) \
          __debugbreak(); \
      } while (0)
#else
#define mDEBUG_BREAK() __builtin_trap()
#endif
#else
#define mDEBUG_BREAK()
#endif

void mDebugOut(const char *format, ...);

#ifdef GIT_BUILD
void mPrintError(char *function, char *file, const int32_t line, const mResult error, const char *expression, const char *commitSHA = GIT_REF);

#define mRESULT_PRINT_FUNCTION_TITLE nullptr
#define mRESULT_PRINT_DEBUG_STRINGIFY(x) nullptr
#define mRESULT_PRINT_DEBUG_STRINGIFY_RETURN_RESULT(result) nullptr
#else
void mPrintError(char *function, char *file, const int32_t line, const mResult error, const char *expression, const char *commitSHA = nullptr);

#define mRESULT_PRINT_FUNCTION_TITLE __FUNCTION__
#define mRESULT_PRINT_DEBUG_STRINGIFY(x) #x
#define mRESULT_PRINT_DEBUG_STRINGIFY_RETURN_RESULT(result) "mRETURN_RESULT(" #result ")"
#endif

mFUNCTION(mResult_ToString, const mResult result, OUT struct mString *pString);

void mDeinit();
void mDeinit(const std::function<void(void)> &param);

template <typename ...Args>
inline void mDeinit(const std::function<void(void)> &param, Args && ...args)
{ 
  if (param) 
    param();

  mDeinit(std::forward<Args>(args)); 
};

#define mSET_ERROR_RAW(resultCode) \
  do \
  { g_mResult_lastErrorResult = resultCode; \
    g_mResult_lastErrorFile = __FILE__; \
    g_mResult_lastErrorLine = __LINE__; \
  } while (0)

#define mRETURN_RESULT(result) \
  do \
  { if (mFAILED(result) && result != mR_Break) \
    { mSET_ERROR_RAW(result); \
      mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __FILE__, __LINE__, result, mRESULT_PRINT_DEBUG_STRINGIFY_RETURN_RESULT(result)); \
      if (g_mResult_breakOnError) \
      { mDEBUG_BREAK(); \
      } \
    } \
    mRESULT_RETURN_RESULT_RAW(result); \
  } while (0)

#define mERROR_CHECK(functionCall, ...) \
  do \
  { mSTDRESULT = (functionCall); \
    if (mFAILED(mSTDRESULT)) \
    { mSET_ERROR_RAW(mSTDRESULT); \
      mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __FILE__, __LINE__, mSTDRESULT, mRESULT_PRINT_DEBUG_STRINGIFY(functionCall)); \
      mDeinit(__VA_ARGS__); \
      if (g_mResult_breakOnError) \
      { mDEBUG_BREAK(); \
      } \
      mRESULT_RETURN_RESULT_RAW(mSTDRESULT); \
    } \
  } while (0)

#define mERROR_IF(conditional, resultOnError, ...) \
  do \
  { if (conditional) \
    { mSTDRESULT = (resultOnError); \
      if (mFAILED(resultOnError)) \
      { mSET_ERROR_RAW(mSTDRESULT); \
        mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __FILE__, __LINE__, mSTDRESULT, mRESULT_PRINT_DEBUG_STRINGIFY(conditional)); \
        mDeinit(__VA_ARGS__); \
        if (g_mResult_breakOnError) \
        { mDEBUG_BREAK(); \
        } \
      } \
      mRESULT_RETURN_RESULT_RAW(mSTDRESULT); \
    } \
  } while (0)

#define mERROR_CHECK_GOTO(functionCall, result, label, ...) \
  do \
  { result = (functionCall); \
    if (mFAILED(result)) \
    { mSET_ERROR_RAW(result); \
      mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __FILE__, __LINE__, result, mRESULT_PRINT_DEBUG_STRINGIFY(functionCall)); \
      mDeinit(__VA_ARGS__); \
      if (g_mResult_breakOnError) \
      { mDEBUG_BREAK(); \
      } \
      goto label; \
    } \
  } while (0)

#define mERROR_IF_GOTO(conditional, resultOnError, result, label, ...) \
  do \
  { if (conditional) \
    { result = (resultOnError); \
      if (mFAILED(resultOnError)) \
      { mSET_ERROR_RAW(result); \
        mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __FILE__, __LINE__, result, mRESULT_PRINT_DEBUG_STRINGIFY(conditional)); \
        mDeinit(__VA_ARGS__); \
        if (g_mResult_breakOnError) \
        { mDEBUG_BREAK(); \
        } \
      } \
      goto label; \
    } \
  } while (0)

#endif // mResult_h__
