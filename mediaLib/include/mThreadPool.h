// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mThreadPool_h__
#define mThreadPool_h__

#include "default.h"
#include "mThreading.h"

struct mTask;

enum mTask_State
{
  mT_S_NotInitialized,
  mT_S_Initialized,
  mT_S_Enqueued,
  mT_S_Running,
  mT_S_Complete,
  mT_S_Aborted,
};

mFUNCTION(mTask_Create, OUT mTask **ppTask, const std::function<mResult(void)> &function);
mFUNCTION(mTask_CreateInplace, IN mTask *pTask, const std::function<mResult(void)> &function);
mFUNCTION(mTask_Destroy, IN_OUT mTask **ppTask);

mFUNCTION(mTask_Join, IN mTask *pTask, const size_t timeoutMilliseconds = mSemaphore_SleepTime::mS_ST_Infinite);
mFUNCTION(mTask_Execute, IN mTask *pTask);
mFUNCTION(mTask_Abort, IN mTask *pTask);

mFUNCTION(mTask_GetResult, IN mTask *pTask, OUT mResult *pResult);
mFUNCTION(mTask_GetState, IN mTask *pTask, OUT mTask_State *pTaskState);

template <class TFunction, class ...Args, class = typename enable_if<!is_same<typename decay<TFunction>::type, thread>::value>::type>
mFUNCTION(mTask_Create, OUT mTask **ppTask, TFunction&& function, Args&&... args)
{
  mTask_Create(ppTask, (std::function<mResult(void)>)
    [=]()
  {
    mResult result = mR_Success;

    mStaticIf<std::is_same<mResult, typename std::decay<function>::type>::value>(
      [&]() { result = function(std::forward<Args>(args)...); })
      .Else([]() { function(std::forward<Args>(args)...); });

    RETURN result;
  });
}

//////////////////////////////////////////////////////////////////////////

struct mThreadPool;

enum mThreadPool_ThreadCount : size_t
{
  mTP_TC_NumberOfLogicalCores = (size_t)-1,
};

mFUNCTION(mThreadPool_Create, OUT mPtr<mThreadPool> *pThreadPool, const size_t threads = mThreadPool_ThreadCount::mTP_TC_NumberOfLogicalCores);
mFUNCTION(mThreadPool_Destroy, IN_OUT mPtr<mThreadPool> *pThreadPool);
mFUNCTION(mThreadPool_EnqueueTask, mPtr<mThreadPool> &threadPool, IN mTask *pTask);

#endif // mThreadPool_h__
