// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mThreadPool.h"

struct mTask
{
  std::function<mResult(void)> function;
  std::atomic<mTask_State> state;
  mSemaphore *pSemaphore;
  mResult result;
};

mFUNCTION(mTask_Destroy_Internal, IN mTask *pTask);

mFUNCTION(mTask_Create, OUT mTask **ppTask, const std::function<mResult(void)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppTask == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAllocZero(ppTask, 1));
  mERROR_CHECK(mTask_CreateInplace(*ppTask, function));

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_CreateInplace, IN mTask *pTask, const std::function<mResult(void)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  pTask->result = mR_Success;
  new (&pTask->function) std::function<mResult(void)>(function);
  new (&pTask->state) std::atomic<mTask_State>(mT_S_Initialized);

  mERROR_CHECK(mSemaphore_Create(&pTask->pSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Destroy, IN_OUT mTask **ppTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppTask == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mTask_Destroy_Internal(*ppTask));

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Join, IN mTask *pTask, const size_t timeoutMilliseconds /* = mSemaphore_SleepTime::mS_ST_Infinite */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  if (pTask->state < mTask_State::mT_S_Complete)
  {
    if (pTask->pSemaphore != nullptr)
    {
      mERROR_CHECK(mSemaphore_Sleep(pTask->pSemaphore, timeoutMilliseconds));
    }
    else
    {
      clock_t start = clock();

      while (true)
      {
        mERROR_IF(timeoutMilliseconds != mSemaphore_SleepTime::mS_ST_Infinite && clock() - start > timeoutMilliseconds, mR_Timeout);

        if (pTask->state > mTask_State::mT_S_Running)
          mRETURN_SUCCESS();
      }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Execute, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mResult result = mR_Success;
  bool hasBeenExecuted = false;
  mDefer<mSemaphore> defer;

  if (pTask->pSemaphore != nullptr)
  {
    mERROR_CHECK_GOTO(mSemaphore_Lock(pTask->pSemaphore), result, epilogue);
    defer = mDefer_Create(mSemaphore_Unlock, pTask->pSemaphore);
  }

  if (pTask->state < mTask_State::mT_S_Running)
    pTask->state = mTask_State::mT_S_Running;
  else
    mRETURN_SUCCESS();

  mERROR_IF_GOTO(pTask->function == nullptr, mR_NotInitialized, result, epilogue);

  pTask->result = pTask->function();
  hasBeenExecuted = true;

  if (pTask->pSemaphore != nullptr)
    mERROR_CHECK_GOTO(mSemaphore_WakeAll(pTask->pSemaphore), result, epilogue);

epilogue:
  if (!hasBeenExecuted || (mSUCCEEDED(pTask->result) && mFAILED(result)))
    pTask->result = result;

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Abort, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  if (pTask->state < mTask_State::mT_S_Running)
  {
    if (pTask->pSemaphore != nullptr)
      mERROR_CHECK(mSemaphore_Lock(pTask->pSemaphore));

    pTask->state = mTask_State::mT_S_Aborted;

    if (pTask->pSemaphore != nullptr)
    {
      mERROR_CHECK(mSemaphore_Unlock(pTask->pSemaphore));
      mERROR_CHECK(mSemaphore_WakeAll(pTask->pSemaphore));
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_GetResult, IN mTask *pTask, OUT mResult *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr || pResult == nullptr, mR_ArgumentNull);

  *pResult = pTask->result;

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_GetState, IN mTask *pTask, OUT mTask_State *pTaskState)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr || pTaskState == nullptr, mR_ArgumentNull);

  *pTaskState = pTask->state;

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Destroy_Internal, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSemaphore_Destroy(&pTask->pSemaphore));
  pTask->state.~atomic();
  pTask->function.~function();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mThreadPool
{
  volatile bool isRunning;
  size_t threadCount;
  mThread **ppThreads;
  mSemaphore *pSemaphore;
};

mFUNCTION(mThreadPool_Create_Internal, mThreadPool *pThreadPool, const size_t threads);
mFUNCTION(mThreadPool_Destroy_Internal, mThreadPool *pThreadPool);

void WorkerThread(mThreadPool *pThreadPool)
{
  mASSERT(pThreadPool != nullptr, "ThreadPool cannot be nullptr.");

  while (pThreadPool->isRunning)
  {

  }
}

mFUNCTION(mThreadPool_Create, OUT mPtr<mThreadPool> *pThreadPool, const size_t threads /* = mThreadPool_ThreadCount::mTP_TC_NumberOfLogicalCores */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThreadPool == nullptr, mR_ArgumentNull);

  if (*pThreadPool != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pThreadPool));
    *pThreadPool = nullptr;
  }

  mThreadPool *pObject = nullptr;
  mDEFER_DESTRUCTION_ON_ERROR(&pObject, mFreePtr);
  mERROR_CHECK(mAllocZero(&pObject, 1));

  mDEFER_DESTRUCTION_ON_ERROR(pThreadPool, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mThreadPool>(pThreadPool, pObject, [](mThreadPool *pData) { mThreadPool_Destroy_Internal(pData); }, mAT_mAlloc));
  pObject = nullptr; // to not be destroyed on error.

  mERROR_CHECK(mThreadPool_Create_Internal(pThreadPool->GetPointer(), threads));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_Destroy, IN_OUT mPtr<mThreadPool> *pThreadPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThreadPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mThreadPool_Destroy_Internal(pThreadPool->GetPointer()));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_EnqueueTask, mPtr<mThreadPool> &threadPool, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTask == nullptr, mR_ArgumentNull);

  // TODO: Enqueue task.

  mERROR_CHECK(mSemaphore_WakeOne(threadPool->pSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_Create_Internal, mThreadPool *pThreadPool, const size_t threads)
{
  mFUNCTION_SETUP();

  if (threads == mThreadPool_ThreadCount::mTP_TC_NumberOfLogicalCores)
  {
    const mResult result = mThread_GetMaximumConcurrentThreads(&pThreadPool->threadCount);

    if (mFAILED(result))
    {
      if (result == mR_OperationNotSupported)
        pThreadPool->threadCount = mThreadPool_ThreadCount::mTP_TC_DefaulThreadCount;
      else
        mRETURN_RESULT(result);
    }
  }
  else
  {
    pThreadPool->threadCount = threads;
  }

  mERROR_CHECK(mAllocZero(&pThreadPool->ppThreads, pThreadPool->threadCount));

  for (size_t i = 0; i < pThreadPool->threadCount; ++i)
    mERROR_CHECK(mThread_Create(&pThreadPool->ppThreads[i], &WorkerThread, pThreadPool));

  // TODO: Create Task queue.

  mERROR_CHECK(mSemaphore_Create(&pThreadPool->pSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_Destroy_Internal, mThreadPool *pThreadPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThreadPool == nullptr, mR_ArgumentNull);

  pThreadPool->isRunning = false;

  // TODO: For each task: abort.

  if (pThreadPool->ppThreads != nullptr)
  {
    for (size_t i = 0; i < pThreadPool->threadCount; ++i)
      if (pThreadPool->ppThreads[i] != nullptr)
        mERROR_CHECK(mThread_Destroy(&pThreadPool->ppThreads[i]));

    mERROR_CHECK(mFreePtr(&pThreadPool->ppThreads));
  }

  pThreadPool->threadCount = 0;

  if(pThreadPool->pSemaphore != nullptr)
    mERROR_CHECK(mSemaphore_Destroy(&pThreadPool->pSemaphore));

  // TODO: delete task queue.

  mRETURN_SUCCESS();
}
