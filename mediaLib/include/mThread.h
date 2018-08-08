// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mThread_h__
#define mThread_h__

#include "default.h"
#include <thread>
#include <atomic>

enum mThread_ThreadState
{
  mT_TS_NotStarted = 0,
  mT_TS_Running,
  mT_TS_Stopped,
};

struct mThread
{
  std::thread handle;
  std::atomic<mThread_ThreadState> threadState;
};

template<class TFunction, class... Args, class = typename enable_if<!is_same<typename decay<TFunction>::type, thread>::value>::type>
void _mThread_ThreadInternalFunc(mThread *pThread, TFunction&& function, Args&&... args)
{
  if(pThread != nullptr)
    *pThread->threadState = mT_TS_Running;

  function(std::forward<Args>(args)...);

  if (pThread != nullptr)
    *pThread->threadState = mT_TS_Stopped;
}

// Creates and starts a thread.
template<class TFunction, class... Args, class = typename enable_if<!is_same<typename decay<TFunction>::type, thread>::value>::type>
mFUNCTION(mThread_Create, OUT mThread **ppThread, TFunction&& function, Args&&... args)
{
  mFUNCTION_SETUP();

  mERROR_IF(pThread == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION_ON_ERROR(ppThread, mSetToNullptr);
  mERROR_CHECK(mAllocZero(ppThread, 1));

  new ((*ppThread)->threadState) mThread_ThreadState(mT_TS_NotStarted);
  new ((*ppThread)->handle) std::thread(_mThread_ThreadInternalFunc, pThread, std::forward<TFunction>(function), std::forward<Args>(args)...);

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_Destroy, IN_OUT mThread **ppThread);
mFUNCTION(mThread_Join, IN mThread *pThread);
mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT std::thread::native_handle_type *pNativeHandle);

#endif // mThread_h__
