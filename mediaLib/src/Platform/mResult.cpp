#include "mediaLib.h"

thread_local const char *g_mResult_lastErrorFile = "<>";
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

void mPrintError(char *function, char *file, const int32_t line, const mResult error, const char *expression)
{
  if (g_mResult_silent)
    return;

#ifdef GIT_BUILD
  mUnused(function, expression);
  mPRINT_ERROR("Error 0x%" PRIx32 " in File '%s' Line % " PRIi32 ".\n", error, file, line);
#else
  const char *expr = "";

  if (expression)
    expr = expression;

  mPRINT_ERROR("Error %s in '%s' (File '%s'; Line % " PRIi32 ") [0x%" PRIx32 "].\nExpression: '%s'.\n\n", mResult_ToString(error), function, file, line, error, expr);
#endif
}

const char * mResult_ToString(const mResult result)
{
#if defined(GIT_BUILD) && !defined(_DEBUG)
  mUnused(result);
  return "!";
#else
#define mRESULT_STRINGIFY_COMMA_SEPARATED(A) #A ,

  static const char * mResultAsString[] = { mRESULT_X_MACRO(mRESULT_STRINGIFY_COMMA_SEPARATED) };

  if (result >= mResult_Count || result < mR_Success)
    return "<Unknown mResult>";
  else
    return mResultAsString[result];
#endif
}

void mDeinit() { }

void mDeinit(const std::function<void(void)> &param)
{
  if (param)
    param();
}
