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

template<class TFunction>
struct _mThread_DataStruct
{
  mThread *pThread; TFunction *pFunction;// std::tuple<Args> args;
};

template<class TFunction>
_mThread_DataStruct<TFunction> _MakeDataStruct(mThread *pThread, TFunction *pFunction)
{
  _mThread_DataStruct<TFunction> dataStruct;
  dataStruct.pFunction = pFunction;
  dataStruct.pThread = pThread;
  return dataStruct;
}

//template<class TFunction, class ...Args>
//void _mThread_ThreadInternalFunc(mThread *pThread, TFunction *pFunction, Args... args)
//{
//  mUnused(pFunction, pThread, std::forward<Args>(args)...);
//  //mThread *pThread, TFunction&& function, Args&&... args;
//
//  if(pThread != nullptr)
//    pThread->threadState = mT_TS_Running;
//  
//  //function(std::forward<Args>(args)...);
//  
//  if (pThread != nullptr)
//    pThread->threadState = mT_TS_Stopped;
//}

template<typename ...Args>
void _mThread_ThreadInternalFunc1(Args... args)
{
  mUnused(std::forward<Args>(args)...);
}

// Creates and starts a thread.
template<class TFunction, class... Args>
mFUNCTION(mThread_Create, OUT mThread **ppThread, TFunction *pFunction, Args&&... args)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppThread == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION_ON_ERROR(ppThread, mSetToNullptr);
  mERROR_CHECK(mAllocZero(ppThread, 1));

  mUnused(pFunction, std::forward<Args>(args)...);
  //auto dataStruct = _MakeDataStruct(*ppThread, pFunction);

  new (&(*ppThread)->threadState) std::atomic<mThread_ThreadState>(mT_TS_NotStarted);
  //new (&(*ppThread)->handle) std::thread(&_mThread_ThreadInternalFunc<TFunction, Args...>, *ppThread, pFunction, args...);
  std::thread t(&_mThread_ThreadInternalFunc1<Args...>, args...);

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_Destroy, IN_OUT mThread **ppThread);
mFUNCTION(mThread_Join, IN mThread *pThread);
mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT std::thread::native_handle_type *pNativeHandle);

mFUNCTION(mThread_GetMaximumConcurrentThreads, OUT size_t *pMaximumConcurrentThreadCount);

#endif // mThread_h__
