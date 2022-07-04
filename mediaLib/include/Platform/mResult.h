#ifndef mResult_h__
#define mResult_h__

#include <functional>
#include <type_traits>
#include <windows.h>
#include <debugapi.h>
#include <stdarg.h>

#define mRESULT_X_MACRO(XX) \
  XX(mR_Success) \
     \
  XX(mR_Break) \
     \
  XX(mR_InvalidParameter) \
  XX(mR_ArgumentNull) \
  XX(mR_InternalError) \
  XX(mR_MemoryAllocationFailure) \
  XX(mR_NotImplemented) \
  XX(mR_NotInitialized) \
  XX(mR_IndexOutOfBounds) \
  XX(mR_ArgumentOutOfBounds) \
  XX(mR_Timeout) \
  XX(mR_OperationNotSupported) \
  XX(mR_ResourceNotFound) \
  XX(mR_ResourceInvalid) \
  XX(mR_ResourceStateInvalid) \
  XX(mR_ResourceIncompatible) \
  XX(mR_EndOfStream) \
  XX(mR_RenderingError) \
  XX(mR_Failure) \
  XX(mR_NotSupported) \
  XX(mR_ResourceAlreadyExists) \
  XX(mR_IOFailure) \
  XX(mR_InsufficientPrivileges) \
  XX(mR_ResourceBusy)

// PLEASE ONLY ADD NEW RESULT TYPES AT THE END OF THIS LIST.
// OTHER APPLICATIONS MAY BE RELYING ON THE CORRESPONDING NUMBERS ASSOCAITED WITH THE PREEXISTING ORDER.

#define mRESULT_SEPARATE_WITH_COMMA(A) A ,

#pragma warning (push)
#pragma warning (disable: 26812)
enum mResult
{
  mRESULT_X_MACRO(mRESULT_SEPARATE_WITH_COMMA)

  mResult_Count
};
#pragma warning (pop)

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

typedef void (OnErrorFunc)(const mResult result);
extern "C" OnErrorFunc *g_mResult_onError;
extern "C" OnErrorFunc *g_mResult_onIndirectError;

extern "C" void mOnError(const mResult error);
extern "C" void mOnIndirectError(const mResult error);

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
      #define mDEBUG_BREAK() do { } while (0)
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
  #define mDEBUG_BREAK() do { } while (0)
#endif

void mDebugOut(const char *text);

template <typename ...Args>
inline void mDebugOut(Args && ... args)
{
  mDebugOut(mFormat(args...));
}

void mPrintError(char *function, char *file, const int32_t line, const mResult error, const char *expression);

#ifdef GIT_BUILD
#define mRESULT_PRINT_FUNCTION_TITLE ""
#define mRESULT_PRINT_DEBUG_STRINGIFY(x) reinterpret_cast<const char *>(nullptr)
#define mRESULT_PRINT_DEBUG_STRINGIFY_RETURN_RESULT(result) reinterpret_cast<const char *>(nullptr)
#else
#define mRESULT_PRINT_FUNCTION_TITLE __FUNCTION__
#define mRESULT_PRINT_DEBUG_STRINGIFY(x) #x
#define mRESULT_PRINT_DEBUG_STRINGIFY_RETURN_RESULT(result) "mRETURN_RESULT(" #result ")"
#endif

const char * mResult_ToString(const mResult result);

inline void mDeinit() {}
inline void mDeinit(const std::function<void(void)> &param) { if (param != nullptr) param(); }

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
    g_mResult_lastErrorFile = __M_FILE__; \
    g_mResult_lastErrorLine = __LINE__; \
  } while (0)

#ifndef GIT_BUILD
#define mDEBUG_BREAK_ON_ERROR() do { if (g_mResult_breakOnError) mDEBUG_BREAK(); } while (0)
#define mDEBUG_BREAK_ON_INDIRECT_ERROR() do { if (g_mResult_breakOnError) mDEBUG_BREAK(); } while (0)
#else
#define mDEBUG_BREAK_ON_ERROR() 
#define mDEBUG_BREAK_ON_INDIRECT_ERROR() 
#endif

#define mRETURN_RESULT(result) \
  do \
  { const mResult _internal_result = (result); \
    if (mFAILED(_internal_result) && _internal_result != mR_Break) \
    { mSET_ERROR_RAW(_internal_result); \
      mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __M_FILE__, __LINE__, _internal_result, mRESULT_PRINT_DEBUG_STRINGIFY_RETURN_RESULT(result)); \
      mOnError(_internal_result); \
      mDEBUG_BREAK_ON_ERROR(); \
    } \
    mRESULT_RETURN_RESULT_RAW(_internal_result); \
  } while (0)

#define mERROR_CHECK(functionCall, ...) \
  do \
  { mSTDRESULT = (functionCall); \
    if (mFAILED(mSTDRESULT)) \
    { mSET_ERROR_RAW(mSTDRESULT); \
      mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __M_FILE__, __LINE__, mSTDRESULT, mRESULT_PRINT_DEBUG_STRINGIFY(functionCall)); \
      mDeinit(__VA_ARGS__); \
      mOnIndirectError(mSTDRESULT); \
      mDEBUG_BREAK_ON_INDIRECT_ERROR(); \
      mRESULT_RETURN_RESULT_RAW(mSTDRESULT); \
    } \
  } while (0)

#define mERROR_IF(conditional, resultOnError, ...) \
  do \
  { if (conditional) \
    { mSTDRESULT = (resultOnError); \
      if (mFAILED(mSTDRESULT) && mSTDRESULT != mR_Break) \
      { mSET_ERROR_RAW(mSTDRESULT); \
        mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __M_FILE__, __LINE__, mSTDRESULT, mRESULT_PRINT_DEBUG_STRINGIFY(conditional)); \
        mDeinit(__VA_ARGS__); \
        mOnError(mSTDRESULT); \
        mDEBUG_BREAK_ON_ERROR(); \
      } \
      mRESULT_RETURN_RESULT_RAW(mSTDRESULT); \
    } \
  } while (0)

#define mERROR_CHECK_GOTO(functionCall, result, label, ...) \
  do \
  { result = (functionCall); \
    if (mFAILED(result)) \
    { mSET_ERROR_RAW(result); \
      mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __M_FILE__, __LINE__, result, mRESULT_PRINT_DEBUG_STRINGIFY(functionCall)); \
      mDeinit(__VA_ARGS__); \
      mOnIndirectError(result); \
      mDEBUG_BREAK_ON_INDIRECT_ERROR(); \
      goto label; \
    } \
  } while (0)

#define mERROR_IF_GOTO(conditional, resultOnError, result, label, ...) \
  do \
  { if (conditional) \
    { result = (resultOnError); \
      if (mFAILED(result) && result != mR_Break) \
      { mSET_ERROR_RAW(result); \
        mPrintError(mRESULT_PRINT_FUNCTION_TITLE, __M_FILE__, __LINE__, result, mRESULT_PRINT_DEBUG_STRINGIFY(conditional)); \
        mDeinit(__VA_ARGS__); \
        mOnError(result); \
        mDEBUG_BREAK_ON_ERROR(); \
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

#ifdef _DEBUG
#define mSILENCE_ERROR_DEBUG(result) mSILENCE_ERROR(result)
#else
#define mSILENCE_ERROR_DEBUG(result) (result)
#endif

#endif // mResult_h__
