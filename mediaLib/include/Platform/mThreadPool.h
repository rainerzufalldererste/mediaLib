#ifndef mThreadPool_h__
#define mThreadPool_h__

#include "mediaLib.h"
#include "mThreading.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "6JvMOAuXIp739fQr49g1bav087hKSpbsPiUkIUX2ZAimpzXy/DszXK+xyq0dEvTqznMYniI/7XKVEivM"
#endif

// `mTask`s are _reference_counted_! No need to worry about leaving tasks behind.
struct mTask;

// Ordered so that every `mTask_State` >= `mT_S_Running` will not be executed and `mTask_State` >= `mT_S_Complete` is considered to be done.
enum mTask_State
{
  mT_S_NotInitialized,
  mT_S_Initialized,
  mT_S_Enqueued,
  mT_S_Running,
  mT_S_Complete,
  mT_S_Aborted,
};

mFUNCTION(mTask_CreateWithLambda, OUT mTask **ppTask, IN OPTIONAL mAllocator *pAllocator, const std::function<mResult(void)> &function);
mFUNCTION(mTask_CreateInplace, IN mTask *pTask, const std::function<mResult(void)> &function);
mFUNCTION(mTask_Destroy, IN_OUT mTask **ppTask);
mFUNCTION(mTask_AddReference, IN mTask *pTask);

mFUNCTION(mTask_Join, IN mTask *pTask, const size_t timeoutMilliseconds = mSemaphore_SleepTime::mS_ST_Infinite);
mFUNCTION(mTask_Execute, IN mTask *pTask);

// Will only abort the task if it can still be aborted. Will not fail if a task can no longer be aborted.
mFUNCTION(mTask_Abort, IN mTask *pTask);

mFUNCTION(mTask_GetResult, IN mTask *pTask, OUT mResult *pResult);
mFUNCTION(mTask_GetState, IN mTask *pTask, OUT mTask_State *pTaskState);

template <class TFunction, class ...Args, class = typename std::enable_if<!std::is_same<typename std::decay<TFunction>::type, std::thread>::value>::type>
mFUNCTION(mTask_Create, OUT mTask **ppTask, TFunction &&function, Args &&...args)
{
  mTask_Create(ppTask, (std::function<mResult(void)>)
    [=]()
  {
    mResult result = mR_Success;

    mStaticIf<std::is_same<mResult, typename std::decay<function>::type>::value> // TODO: This should use the same logic as mThread!
      ([&]() { result = function(std::forward<Args>(args)...); })
      .Else([]() { function(std::forward<Args>(args)...); });

    RETURN result;
  });
}

//////////////////////////////////////////////////////////////////////////

struct mThreadPool;

enum mThreadPool_ThreadCount : size_t
{
  mTP_TC_NumberOfLogicalCores = (size_t)-1,
  mTP_TC_DefaulThreadCount = 4,
};

mFUNCTION(mThreadPool_Create, OUT mPtr<mThreadPool> *pThreadPool, IN OPTIONAL mAllocator *pAllocator, const size_t threads = mThreadPool_ThreadCount::mTP_TC_NumberOfLogicalCores);
mFUNCTION(mThreadPool_Destroy, IN_OUT mPtr<mThreadPool> *pThreadPool);
mFUNCTION(mThreadPool_Clear, mPtr<mThreadPool> &asyncTaskHandler);
mFUNCTION(mThreadPool_EnqueueTask, mPtr<mThreadPool> &asyncTaskHandler, IN mTask *pTask);
mFUNCTION(mThreadPool_GetThreadCount, mPtr<mThreadPool> &asyncTaskHandler, OUT size_t *pThreadCount);

//////////////////////////////////////////////////////////////////////////

// This provides the same functionality and performance from htCodec's thread pool.
struct mTasklessThreadPool;

mFUNCTION(mTasklessThreadPool_Create, OUT mPtr<mTasklessThreadPool> *pThreadPool, IN mAllocator *pAllocator, const size_t threadCount);
mFUNCTION(mTasklessThreadPool_Destroy, IN_OUT mPtr<mTasklessThreadPool> *pThreadPool);

mFUNCTION(mTasklessThreadPool_EnqueueTask, mPtr<mTasklessThreadPool> &threadPool, const std::function<void(void)> &taskHandle);
mFUNCTION(mTasklessThreadPool_WaitForAll, mPtr<mTasklessThreadPool> &threadPool);

mFUNCTION(htCodecThreadPool_GetWorkerThreadCount, const mPtr<mTasklessThreadPool> &threadPool, OUT size_t *pCount);

#endif // mThreadPool_h__
