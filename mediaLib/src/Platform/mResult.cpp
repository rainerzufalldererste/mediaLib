#include "mediaLib.h"

thread_local const char *g_mResult_lastErrorFile = nullptr;
thread_local size_t g_mResult_lastErrorLine = 0;
thread_local mResult g_mResult_lastErrorResult = mR_Success;

#if !defined(GIT_BUILD)
bool g_mResult_breakOnError_default = false;
thread_local bool g_mResult_breakOnError = g_mResult_breakOnError_default;
#endif

bool g_mResult_silent_default = false;
thread_local bool g_mResult_silent = g_mResult_silent_default;

void mDebugOut(const char *format, ...)
{
#if !defined(GIT_BUILD)
  char buffer[1024 * 16];

  mMemset(buffer, mARRAYSIZE(buffer), 0);

  va_list args;
  va_start(args, format);
  vsprintf_s(buffer, format, args);
  va_end(args);

  buffer[mARRAYSIZE(buffer) - 1] = 0;

  if (buffer[0] != '\0')
    OutputDebugStringA(buffer);
#else
  mUnused(format);
#endif
}

void mPrintError(char *function, char *file, const int32_t line, const mResult error, const char *expression, const char *commitSHA /* = nullptr */)
{
  if (g_mResult_silent)
    return;

#ifndef GIT_BUILD
  mString errorName;
#endif

  const char *expr = "";
  const char *buildID = "";

  if (expression)
    expr = expression;

  if (commitSHA)
    buildID = commitSHA;

#ifdef GIT_BUILD
  mUnused(function);

  mPRINT_ERROR("Error 0x%" PRIx32 " in File '%s' Line % " PRIi32 ". ('%s')\n", error, file, line, buildID);
#else
  mUnused(buildID);

  if (mFAILED(mResult_ToString(error, &errorName)))
    mPRINT_ERROR("Error in '%s' (File '%s'; Line % " PRIi32 ") [0x%" PRIx32 "].\nExpression: '%s'.\n\n", function, file, line, error, expr);
  else
    mPRINT_ERROR("Error %s in '%s' (File '%s'; Line % " PRIi32 ") [0x%" PRIx32 "].\nExpression: '%s'.\n\n", errorName.c_str(), function, file, line, error, expr);
#endif
}

mFUNCTION(mResult_ToString, const mResult result, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_InternalError);

  switch (result)
  {
  case mR_Success:
    *pString = "mR_Success";
    break;

  case mR_Break:
    *pString = "mR_Break";
    break;

  case mR_InvalidParameter:
    *pString = "mR_InvalidParameter";
    break;

  case mR_ArgumentNull:
    *pString = "mR_ArgumentNull";
    break;

  case mR_InternalError:
    *pString = "mR_InternalError";
    break;

  case mR_MemoryAllocationFailure:
    *pString = "mR_MemoryAllocationFailure";
    break;

  case mR_NotImplemented:
    *pString = "mR_NotImplemented";
    break;

  case mR_NotInitialized:
    *pString = "mR_NotInitialized";
    break;

  case mR_IndexOutOfBounds:
    *pString = "mR_IndexOutOfBounds";
    break;

  case mR_ArgumentOutOfBounds:
    *pString = "mR_ArgumentOutOfBounds";
    break;

  case mR_Timeout:
    *pString = "mR_Timeout";
    break;

  case mR_OperationNotSupported:
    *pString = "mR_OperationNotSupported";
    break;

  case mR_ResourceNotFound:
    *pString = "mR_ResourceNotFound";
    break;

  case mR_ResourceInvalid:
    *pString = "mR_ResourceInvalid";
    break;

  case mR_ResourceStateInvalid:
    *pString = "mR_ResourceStateInvalid";
    break;

  case mR_ResourceIncompatible:
    *pString = "mR_ResourceIncompatible";
    break;

  case mR_EndOfStream:
    *pString = "mR_EndOfStream";
    break;

  case mR_RenderingError:
    *pString = "mR_RenderingError";
    break;

  case mR_Failure:
    *pString = "mR_Failure";
    break;

  case mR_NotSupported:
    *pString = "mR_NotSupported";
    break;

  case mR_ResourceAlreadyExists:
    *pString = "mR_ResourceAlreadyExists";
    break;

  case mR_IOFailure:
    *pString = "mR_IOFailure";
    break;

  case mR_InsufficientPrivileges:
    *pString = "mR_InsufficientPrivileges";
    break;

  default:
    *pString = "<Unknown mResult>";
    break;
  }

  mRETURN_SUCCESS();
}

void mDeinit() { }

void mDeinit(const std::function<void(void)> &param)
{
  if (param)
    param();
}
