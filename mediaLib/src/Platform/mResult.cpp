#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "NqGDAw59sUxsxOtY8VxR3PAuvRnWsQlV14qGs1hgUIRrTgNYkHCuoCq6wS2qUoOeNxV5pf82Lnu7HMpr"
#endif

thread_local const char *g_mResult_lastErrorFile = "<>";
thread_local size_t g_mResult_lastErrorLine = 0;
thread_local mResult g_mResult_lastErrorResult = mR_Success;

#if !defined(GIT_BUILD)
bool g_mResult_breakOnError_default = false;
thread_local bool g_mResult_breakOnError = g_mResult_breakOnError_default;
#endif

bool g_mResult_silent_default = false;
thread_local bool g_mResult_silent = g_mResult_silent_default;

extern "C" OnErrorFunc *g_mResult_onError = nullptr;
extern "C" OnErrorFunc *g_mResult_onIndirectError = nullptr;

void mDebugOut(const char *text)
{
#if !defined(GIT_BUILD)
  if (text != nullptr && text[0] != '\0')
    OutputDebugStringA(text);
#else
  mUnused(text);
#endif
}

void mPrintError(char *function, char *file, const int32_t line, const mResult error, const char *expression)
{
  if (g_mResult_silent)
    return;

#ifdef GIT_BUILD
  mUnused(function, expression);
  mPRINT_ERROR("Error 0x", mFUInt<mFHex>(error), " in File '", file, "' Line ", line, ".");
#else
  const char *expr = "";

  if (expression)
    expr = expression;

  mPRINT_ERROR("Error ", mResult_ToString(error), " in '", function, "' (File '", file, "'; Line ", line, ") [0x", mFUInt<mFHex>(error), "].\nExpression: '", expr, "'.");
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

void mOnError(const mResult error)
{
#ifndef GIT_BUILD
  if (g_mResult_breakOnError)
    mDEBUG_BREAK();
#endif

  if (g_mResult_onError != nullptr)
    g_mResult_onError(error);
}

void mOnIndirectError(const mResult error)
{
#ifndef GIT_BUILD
  if (g_mResult_breakOnError)
    mDEBUG_BREAK();
#endif

  if (g_mResult_onIndirectError != nullptr)
    g_mResult_onIndirectError(error);
}
