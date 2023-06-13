#include "mThread.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Hfwj2y5HffPq7tqTRppX4cjrh5tkCiuFftRWKArZIdsb8g/a2hVBypy8RnxHwnSE19MrRWEh4vEq5JeP"
#endif

static mFUNCTION(mThread_Destroy_Internal, IN_OUT mThread *pThread);

#if defined (mPLATFORM_WINDOWS)
static mFUNCTION(mThread_Priority_ToWinPriority_Internal, const mThread_Priority priority, OUT int32_t *pPriority);
#endif

//////////////////////////////////////////////////////////////////////////

#if defined (mPLATFORM_WINDOWS)
DWORD _mThread_ThreadFuncInternal(void *pThreadParam)
{
  mThread *pThread = reinterpret_cast<mThread *>(pThreadParam);

  mASSERT(pThread != nullptr, "pThread cannot be nullptr.");

  pThread->threadFunc(pThread);

  return (DWORD)pThread->result;
}

mFUNCTION(_mThread_CreateHandleInternal, IN mThread *pThread)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr, mR_ArgumentNull);
  
  pThread->handle = CreateThread(NULL, 0, &_mThread_ThreadFuncInternal, pThread, 0, NULL);
  mERROR_IF(pThread->handle == NULL, mR_InternalError);

  mRETURN_SUCCESS();
}
#endif

mFUNCTION(mThread_Destroy, IN_OUT mThread **ppThread)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppThread == nullptr, mR_ArgumentNull);

  mDEFER_CALL(ppThread, mSetToNullptr);
  mERROR_CHECK(mThread_Destroy_Internal(*ppThread));
  
  mAllocator *pAllocator = (*ppThread)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppThread));

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_Join, IN mThread *pThread)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  const DWORD result = WaitForSingleObject(pThread->handle, INFINITE);
  mERROR_IF(result != WAIT_OBJECT_0, mR_InternalError);
#else
  mERROR_IF(pThread->handle.native_handle() == nullptr, mR_Success);
  mERROR_IF(!pThread->handle.joinable(), mR_OperationNotSupported);

  pThread->handle.join();
#endif

  mRETURN_SUCCESS();
}

#if defined(mPLATFORM_WINDOWS)
mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT HANDLE *pNativeHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr || pNativeHandle == nullptr, mR_ArgumentNull);

  *pNativeHandle = pThread->handle;

  mRETURN_SUCCESS();
}
#else
mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT std::thread::native_handle_type *pNativeHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr || pNativeHandle == nullptr, mR_ArgumentNull);

  *pNativeHandle = pThread->handle.native_handle();

  mRETURN_SUCCESS();
}
#endif

mFUNCTION(mThread_GetMaximumConcurrentThreads, OUT size_t *pMaximumConcurrentThreadCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMaximumConcurrentThreadCount == nullptr, mR_ArgumentNull);

  *pMaximumConcurrentThreadCount = std::thread::hardware_concurrency();

  mERROR_IF(*pMaximumConcurrentThreadCount == 0, mR_OperationNotSupported);

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_SetThreadPriority, IN mThread *pThread, const mThread_Priority priority)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr, mR_ArgumentNull);

#if defined (mPLATFORM_WINDOWS)
  int32_t prio = 0;
  mERROR_CHECK(mThread_Priority_ToWinPriority_Internal(priority, &prio));

  SetThreadPriority(pThread->handle, prio);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_SetCurrentThreadPriority, const mThread_Priority priority)
{
  mFUNCTION_SETUP();

#if defined (mPLATFORM_WINDOWS)
  int32_t prio = 0;
  mERROR_CHECK(mThread_Priority_ToWinPriority_Internal(priority, &prio));

  SetThreadPriority(GetCurrentThread(), prio);
#endif

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mThread_Destroy_Internal, IN_OUT mThread *pThread)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr, mR_ArgumentNull);

#if defined (mPLATFORM_WINDOWS)
  if (pThread->handle != NULL)
  {
    const DWORD result = WaitForSingleObject(pThread->handle, INFINITE);

    if (result != WAIT_OBJECT_0)
      TerminateThread(pThread->handle, 0xDEADF00D);

    CloseHandle(pThread->handle);
  }

  pThread->threadFunc.~function();
#else
  pThread->handle.~thread();
#endif
  pThread->threadState.~atomic();

  mRETURN_SUCCESS();
}

#if defined (mPLATFORM_WINDOWS)
static mFUNCTION(mThread_Priority_ToWinPriority_Internal, const mThread_Priority priority, OUT int32_t *pPriority)
{
  mFUNCTION_SETUP();

  switch (priority)
  {
  case mT_P_Low:
    *pPriority = THREAD_PRIORITY_LOWEST;
    break;

  case mT_P_Normal:
    *pPriority = THREAD_PRIORITY_NORMAL;
    break;

  case mT_P_High:
    *pPriority = THREAD_PRIORITY_HIGHEST;
    break;

  case mT_P_Realtime:
    *pPriority = THREAD_PRIORITY_TIME_CRITICAL;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}
#endif
