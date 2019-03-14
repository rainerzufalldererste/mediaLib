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
  mR_InsufficientPrivileges,

  mResult_Count
};

#define mSTDRESULT __result
#define mSUCCEEDED(result) ((result) == mR_Success)
#define mFAILED(result) (!mSUCCEEDED(result))

#define mFUNCTION(FunctionName, ...) mResult FunctionName(__VA_ARGS__)
#define mFUNCTION_SETUP() mResult mSTDRESULT = mR_Success; mUnused(mSTDRESULT)
#define mRESULT_RETURN_RESULT_RAW(result) do { mSTDRESULT = result; return mSTDRESULT; } while (0)
#define mRETURN_SUCCESS() mRESULT_RETURN_RESULT_RAW(mR_Success)

extern thread_local const char *g_mResult_lastErrorFile;
extern thread_local size_t g_mResult_lastErrorLine;
extern thread_local mResult g_mResult_lastErrorResult;

#if defined(GIT_BUILD)
constexpr bool g_mResult_breakOnError = false;
#else
extern bool g_mResult_breakOnError_default;
extern thread_local bool g_mResult_breakOnError;
#endif

extern bool g_mResult_silent_default;
extern thread_local bool g_mResult_silent;

#ifdef _DEBUG
#define mBREAK_ON_FAILURE 1
#else
#define mBREAK_ON_FAILURE 0
#endif // _DEBUG

#if mBREAK_ON_FAILURE
  #ifdef mPLATFORM_WINDOWS
    #ifdef GIT_BUILD
      #define mDEBUG_BREAK()
    #else
      #define mDEBUG_BREAK() \
        do \
        { BOOL __isRemoteDebuggerPresent__ = false; \
          if (!CheckRemoteDebuggerPresent(GetCurrentProcess(), &__isRemoteDebuggerPresent__)) \
            __isRemoteDebuggerPresent__ = false; \
          if (IsDebuggerPresent() || __isRemoteDebuggerPresent__) \
            __debugbreak(); \
        } while (0)
    #endif
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

class mErrorPushSilent_Internal
{
private:
  const bool previouslySilent
#ifndef GIT_BUILD
    , previouslyBreaking
#endif
    ;

public:
  mErrorPushSilent_Internal() :
    previouslySilent(g_mResult_silent)
#ifndef GIT_BUILD
    ,
    previouslyBreaking(g_mResult_breakOnError)
#endif
  {
    if (!previouslySilent)
      g_mResult_silent = true;

#ifndef GIT_BUILD
    if (previouslyBreaking)
      g_mResult_breakOnError = false;
#endif
  };

  ~mErrorPushSilent_Internal()
  {
    if (!previouslySilent)
      g_mResult_silent = false;

#ifndef GIT_BUILD
    if (previouslyBreaking)
      g_mResult_breakOnError = true;
#endif
  }
};

#define mSILENCE_ERROR(result) (mErrorPushSilent_Internal(), result)

#endif // mResult_h__
