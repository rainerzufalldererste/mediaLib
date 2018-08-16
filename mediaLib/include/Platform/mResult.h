// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mResult_h__
#define mResult_h__

#include <functional>
#include <type_traits>

enum mResult
{
  mR_Success,

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

  mResult_Count
};

#define mSTDRESULT __result
#define mSUCCEEDED(result) ((result) == mR_Success)
#define mFAILED(result) (!mSUCCEEDED(result))

#define mFUNCTION(FunctionName, ...) mResult FunctionName(__VA_ARGS__)
#define mFUNCTION_SETUP() mResult mSTDRESULT = mR_Success
#define mRETURN_RESULT(result) do { mSTDRESULT = result; return mSTDRESULT; } while (0)
#define mRETURN_SUCCESS() mRETURN_RESULT(mR_Success)

extern const char *g_mResult_lastErrorFile;
extern size_t g_mResult_lastErrorLine;
extern mResult g_mResult_lastErrorResult;

extern bool g_mResult_breakOnError;

#ifdef _DEBUG
#define mBREAK_ON_FAILURE true
#else
#define mBREAK_ON_FAILURE false
#endif // _DEBUG

void mDeinit();
void mDeinit(const std::function<void(void)> &param);
template <typename ...Args> void mDeinit(const std::function<void(void)> &param, Args && ...args) { if (param) { param(); } mDeinit(std::forward<Args>(args)); };

#define mERROR_CHECK(functionCall, ...) \
  do \
  { mSTDRESULT = (functionCall); \
    if (mFAILED(mSTDRESULT)) \
    { g_mResult_lastErrorResult = mSTDRESULT; \
      g_mResult_lastErrorFile = __FILE__; \
      g_mResult_lastErrorLine = __LINE__; \
      mDeinit(__VA_ARGS__); \
      if (g_mResult_breakOnError && mBREAK_ON_FAILURE) \
      { __debugbreak(); \
      } \
      mRETURN_RESULT(mSTDRESULT); \
    } \
  } while (0)

#define mERROR_IF(conditional, resultOnError, ...) \
  do \
  { if (conditional) \
    { mSTDRESULT = (resultOnError); \
      if (mFAILED(mSTDRESULT)) \
      { g_mResult_lastErrorResult = mSTDRESULT; \
        g_mResult_lastErrorFile = __FILE__; \
        g_mResult_lastErrorLine = __LINE__; \
        mDeinit(__VA_ARGS__); \
        if (g_mResult_breakOnError && mBREAK_ON_FAILURE) \
        { __debugbreak(); \
        } \
        mRETURN_RESULT(mSTDRESULT); \
      } \
    } \
  } while (0)

#define mERROR_CHECK_GOTO(functionCall, result, label, ...) \
  do \
  { result = (functionCall); \
    if (mFAILED(result)) \
    { g_mResult_lastErrorResult = result; \
      g_mResult_lastErrorFile = __FILE__; \
      g_mResult_lastErrorLine = __LINE__; \
      mDeinit(__VA_ARGS__); \
      if (g_mResult_breakOnError && mBREAK_ON_FAILURE) \
      { __debugbreak(); \
      } \
      goto label; \
    } \
  } while (0)

#define mERROR_IF_GOTO(conditional, resultOnError, result, label, ...) \
  do \
  { if (conditional) \
    { result = (resultOnError); \
      if (mFAILED(result)) \
      { g_mResult_lastErrorResult = result; \
        g_mResult_lastErrorFile = __FILE__; \
        g_mResult_lastErrorLine = __LINE__; \
        mDeinit(__VA_ARGS__); \
        if (g_mResult_breakOnError && mBREAK_ON_FAILURE) \
        { __debugbreak(); \
        } \
        goto label; \
      } \
    } \
  } while (0)


#endif // mResult_h__
