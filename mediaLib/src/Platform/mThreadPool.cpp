#include "mThreadPool.h"
#include "mQueue.h"

struct mTask
{
  std::function<mResult(void)> function;
  std::atomic<mTask_State> state;
  std::atomic<size_t> referenceCount;
  mSemaphore *pSemaphore = nullptr;
  mResult result;
  mAllocator *pAllocator = nullptr;
  bool isAllocated = false;
};

mFUNCTION(mTask_Destroy_Internal, IN mTask *pTask);

mFUNCTION(mTask_CreateWithLambda, OUT mTask **ppTask, IN OPTIONAL mAllocator *pAllocator, const std::function<mResult(void)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppTask == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppTask, 1));
  (*ppTask)->isAllocated = true;
  (*ppTask)->pAllocator = pAllocator;

  mDEFER_CALL_ON_ERROR(ppTask, mTask_Destroy);
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
  new (&pTask->referenceCount) std::atomic<size_t>(1);

  mERROR_CHECK(mSemaphore_Create(&pTask->pSemaphore, pTask->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mTask_Destroy, IN_OUT mTask **ppTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppTask == nullptr || *ppTask == nullptr, mR_ArgumentNull);

  const size_t referenceCount = --(*ppTask)->referenceCount;

  if (referenceCount == 0)
  {
    mAllocator *pAllocator = (*ppTask)->pAllocator;
    const bool wasAllocated = (*ppTask)->isAllocated;

    mERROR_CHECK(mTask_Destroy_Internal(*ppTask));

    if (wasAllocated)
      mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppTask));
  }

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
      for (size_t i = 0; i < timeoutMilliseconds; ++i)
      {
        const mResult result = mSemaphore_Sleep(pTask->pSemaphore, 1);

        if (mSUCCEEDED(result))
        {
          mRETURN_SUCCESS();
        }
        else
        {
          mERROR_IF(result != mR_Timeout, result);

          if (pTask->state < mTask_State::mT_S_Complete)
          {
            if (timeoutMilliseconds == mSemaphore_SleepTime::mS_ST_Infinite)
              i = 0;

            continue;
          }
          else
          {
            mRETURN_SUCCESS();
          }
        }

      }
    }
    else
    {
      clock_t start = clock();

      while (true)
      {
        mERROR_IF(timeoutMilliseconds != mSemaphore_SleepTime::mS_ST_Infinite && clock() - start > timeoutMilliseconds, mR_Timeout);

        if (pTask->state > mTask_State::mT_S_Running)
          mRETURN_SUCCESS();

        mERROR_CHECK(mSleep(1));
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

  {
    mDefer<mSemaphore *> defer;

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

  epilogue:
    if (!hasBeenExecuted || (mSUCCEEDED(pTask->result) && mFAILED(result)))
      pTask->result = result;

    pTask->state = mTask_State::mT_S_Complete;
  }

  result = mSemaphore_WakeAll(pTask->pSemaphore);

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

  if(pTask->pSemaphore)
    mERROR_CHECK(mSemaphore_Destroy(&pTask->pSemaphore));

  pTask->referenceCount.~atomic();
  pTask->state.~atomic();
  pTask->function.~function();
  pTask->pAllocator = nullptr;
  pTask->result = mR_NotInitialized;
  pTask->isAllocated = false;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mThreadPool
{
  volatile bool isRunning;
  size_t threadCount;
  mThread **ppThreads;
  mSemaphore *pSemaphore;
  mAllocator *pAllocator;
  mPtr<mQueue<mTask *>> queue;
};

mFUNCTION(mThreadPool_Create_Internal, mThreadPool *pThreadPool, IN OPTIONAL mAllocator *pAllocator, const size_t threads);
mFUNCTION(mThreadPool_Destroy_Internal, mThreadPool *pThreadPool);

void mThreadPool_WorkerThread(mThreadPool *pThreadPool)
{
  mASSERT(pThreadPool != nullptr, "ThreadPool cannot be nullptr.");

  while (pThreadPool->isRunning)
  {
    mTask *pTask = nullptr;

    {
      mASSERT(mSUCCEEDED(mSemaphore_Lock(pThreadPool->pSemaphore)), "Error in " __FUNCTION__ ": Semaphore error.");
      mDEFER_CALL(pThreadPool->pSemaphore, mSemaphore_Unlock);

      size_t count = 0;
      mASSERT(mSUCCEEDED(mQueue_GetCount(pThreadPool->queue, &count)), "Error in " __FUNCTION__ ": Could not get task queue length.");

      if (count > 0)
        mASSERT(mSUCCEEDED(mQueue_PopFront(pThreadPool->queue, &pTask)), "Error in " __FUNCTION__ ": Could not dequeue task.");
    }

    if (pTask != nullptr)
    {
      mASSERT(mSUCCEEDED(mTask_Execute(pTask)), "Error in " __FUNCTION__ ": Failed to execute task.");
      mASSERT(mSUCCEEDED(mTask_Destroy(&pTask)), "Error in " __FUNCTION__ ": Failed to destroy task.");
    }

    const mResult result = mSemaphore_Sleep(pThreadPool->pSemaphore, 1);

    if (result == mR_Timeout)
      continue;
    else
      mASSERT(mSUCCEEDED(result), "Error in " __FUNCTION__ ": Semaphore error.");
  }
}

mFUNCTION(mThreadPool_Create, OUT mPtr<mThreadPool> *pThreadPool, IN OPTIONAL mAllocator *pAllocator, const size_t threads /* = mThreadPool_ThreadCount::mTP_TC_NumberOfLogicalCores */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThreadPool == nullptr, mR_ArgumentNull);

  if (*pThreadPool != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pThreadPool));
    *pThreadPool = nullptr;
  }

  mThreadPool *pObject = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pObject));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pObject, 1));

  mDEFER_CALL_ON_ERROR(pThreadPool, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mThreadPool>(pThreadPool, pObject, [](mThreadPool *pData) { mThreadPool_Destroy_Internal(pData); }, pAllocator));
  pObject = nullptr; // to not be destroyed on error.

  mERROR_CHECK(mThreadPool_Create_Internal(pThreadPool->GetPointer(), pAllocator, threads));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_Destroy, IN_OUT mPtr<mThreadPool> *pThreadPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThreadPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pThreadPool));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_EnqueueTask, mPtr<mThreadPool> &threadPool, IN mTask *pTask)
{
  mFUNCTION_SETUP();

  mERROR_IF(threadPool == nullptr || pTask == nullptr, mR_ArgumentNull);

  // Enqueue task.
  {
    ++pTask->referenceCount;

    mERROR_CHECK(mSemaphore_Lock(threadPool->pSemaphore));
    mDEFER_CALL(threadPool->pSemaphore, mSemaphore_Unlock);
    mERROR_CHECK(mQueue_PushBack(threadPool->queue, pTask));
  }

  mERROR_CHECK(mSemaphore_WakeOne(threadPool->pSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_GetThreadCount, mPtr<mThreadPool> &threadPool, OUT size_t *pThreadCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(threadPool == nullptr || pThreadCount == nullptr, mR_ArgumentNull);

  *pThreadCount = threadPool->threadCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_Create_Internal, mThreadPool *pThreadPool, IN OPTIONAL mAllocator *pAllocator, const size_t threads)
{
  mFUNCTION_SETUP();

  pThreadPool->pAllocator = pAllocator;
  pThreadPool->isRunning = true;

  mDEFER_CALL_ON_ERROR(&pThreadPool->queue, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&pThreadPool->queue, pThreadPool->pAllocator));

  mDEFER_CALL_ON_ERROR(&pThreadPool->pSemaphore, mSemaphore_Destroy);
  mERROR_CHECK(mSemaphore_Create(&pThreadPool->pSemaphore, pThreadPool->pAllocator));

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

  mERROR_IF(pThreadPool->threadCount == 0, mR_InvalidParameter);

  mERROR_CHECK(mAllocator_AllocateZero(pThreadPool->pAllocator, &pThreadPool->ppThreads, pThreadPool->threadCount));

  for (size_t i = 0; i < pThreadPool->threadCount; ++i)
    mERROR_CHECK(mThread_Create(&pThreadPool->ppThreads[i], pThreadPool->pAllocator, &mThreadPool_WorkerThread, pThreadPool));

  mRETURN_SUCCESS();
}

mFUNCTION(mThreadPool_Destroy_Internal, mThreadPool *pThreadPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThreadPool == nullptr, mR_ArgumentNull);

  pThreadPool->isRunning = false;

  // Remove all tasks.
  {
    mERROR_CHECK(mSemaphore_Lock(pThreadPool->pSemaphore));
    mDEFER_CALL(pThreadPool->pSemaphore, mSemaphore_Unlock);

    size_t count = 0;
    mERROR_CHECK(mQueue_GetCount(pThreadPool->queue, &count));

    for (size_t i = 0; i < count; ++i)
    {
      mTask *pTask = nullptr;

      mERROR_CHECK(mQueue_PopFront(pThreadPool->queue, &pTask));
      mERROR_CHECK(mTask_Destroy(&pTask));
    }
  }

  if (pThreadPool->ppThreads != nullptr)
  {
    for (size_t i = 0; i < pThreadPool->threadCount; ++i)
    {
      if (pThreadPool->ppThreads[i] != nullptr)
      {
        mERROR_CHECK(mThread_Join(pThreadPool->ppThreads[i]));
        mERROR_CHECK(mThread_Destroy(&pThreadPool->ppThreads[i]));
      }
    }

    mERROR_CHECK(mAllocator_FreePtr(pThreadPool->pAllocator, &pThreadPool->ppThreads));
  }

  pThreadPool->threadCount = 0;

  if(pThreadPool->pSemaphore != nullptr)
    mERROR_CHECK(mSemaphore_Destroy(&pThreadPool->pSemaphore));

  if (pThreadPool->queue != nullptr)
    mERROR_CHECK(mQueue_Destroy(&pThreadPool->queue));

  mRETURN_SUCCESS();
}
