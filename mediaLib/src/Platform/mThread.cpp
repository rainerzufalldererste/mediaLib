#include "mThread.h"

mFUNCTION(mThread_Destroy_Internal, IN_OUT mThread *pThread);

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
  mERROR_IF(pThread->handle.native_handle() == nullptr, mR_Success);
  mERROR_IF(!pThread->handle.joinable(), mR_OperationNotSupported);

  pThread->handle.join();

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT std::thread::native_handle_type *pNativeHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr || pNativeHandle == nullptr, mR_ArgumentNull);

  *pNativeHandle = pThread->handle.native_handle();

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_GetMaximumConcurrentThreads, OUT size_t *pMaximumConcurrentThreadCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMaximumConcurrentThreadCount == nullptr, mR_ArgumentNull);

  *pMaximumConcurrentThreadCount = std::thread::hardware_concurrency();

  mERROR_IF(*pMaximumConcurrentThreadCount == 0, mR_OperationNotSupported);

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_Destroy_Internal, IN_OUT mThread *pThread)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr, mR_ArgumentNull);

  pThread->handle.~thread();
  pThread->threadState.~atomic();

  mRETURN_SUCCESS();
}
