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

mFUNCTION(mTask_Create, OUT mTask **ppTask, const std::function<void(void)> &function);
mFUNCTION(mTask_Destroy, IN_OUT mTask **ppTask);

mFUNCTION(mTask_Join, IN mTask *pTask, size_t timeoutMilliseconds = mSemaphore_SleepTime::mS_ST_Infinite);
mFUNCTION(mTask_Abort, IN mTask *pTask);

template <class TFunction, class ...Args, class = typename enable_if<!is_same<typename decay<TFunction>::type, thread>::value>::type>
mFUNCTION(mTask_Create, OUT mTask **ppTask, TFunction&& function, Args&&... args)
{
  mTask_Create(ppTask, [&]() { function(std::forward<Args>(args)...); });
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
