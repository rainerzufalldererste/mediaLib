#include "mResult.h"
#include "mString.h"

const char *g_mResult_lastErrorFile = nullptr;
size_t g_mResult_lastErrorLine = 0;
mResult g_mResult_lastErrorResult = mR_Success;

bool g_mResult_breakOnError = false;

void mDebugOut(const char *format, ...)
{
#if !defined(FINAL)
  char buffer[1024 * 2];

  mMemset(buffer, mARRAYSIZE(buffer), 0);

  va_list args;
  va_start(args, format);
  vsprintf_s(buffer, format, args);
  va_end(args);

  buffer[mARRAYSIZE(buffer) - 1] = 0;

  mTRACE(buffer);
  OutputDebugStringA(buffer);
#else
  mUnused(format);
#endif
}

void mPrintError(char *function, char *file, const int32_t line, const mResult error, const char *expression)
{
  mString errorName;

  const char *expr = "";

  if (expression)
    expr = expression;

  if (mFAILED(mResult_ToString(error, &errorName)))
    mDebugOut("Error in '%s' (File '%s'; Line % " PRIi32 ") [0x%" PRIx32 "].\nExpression: '%s'.\n\n", function, file, line, error, expr);
  else
    mDebugOut("Error %s in '%s' (File '%s'; Line % " PRIi32 ") [0x%" PRIx32 "].\nExpression: '%s'.\n\n", errorName.c_str(), function, file, line, error, expr);
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
