// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mThread.h"

mFUNCTION(mThread_Destroy_Internal, IN_OUT mThread *pThread);

mFUNCTION(mThread_Destroy, IN_OUT mThread **ppThread)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppThread == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION(ppThread, mSetToNullptr);
  mERROR_CHECK(mThread_Destroy_Internal(*ppThread));
  mERROR_CHECK(mFreePtr(ppThread));

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_Join, IN mThread *pThread)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr, mR_ArgumentNull);
  mERROR_IF(pThread->handle.joinable() == false, mR_OperationNotSupported);

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
