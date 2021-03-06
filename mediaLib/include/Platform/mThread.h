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
  mAllocator *pAllocator;
  std::thread handle;
  std::atomic<mThread_ThreadState> threadState;
  mResult result;
};

template<size_t ...>
struct _mThread_ThreadInternalSequence { };

template<size_t TCount, size_t ...TCountArgs>
struct _mThread_ThreadInternalIntegerSequenceGenerator : _mThread_ThreadInternalIntegerSequenceGenerator<TCount - 1, TCount - 1, TCountArgs...> { };

template<size_t ...TCountArgs>
struct _mThread_ThreadInternalIntegerSequenceGenerator<0, TCountArgs...>
{
  typedef _mThread_ThreadInternalSequence<TCountArgs...> type;
};

template<class TFunctionDecay>
struct _mThread_ThreadInternal_Decay
{
  template <typename TFunction, typename ...Args>
  static mResult CallFunction(TFunction function, Args&&...args)
  {
    function(std::forward<Args>(args)...);

    return mR_Success;
  }
};

template <>
struct _mThread_ThreadInternal_Decay<mResult>
{
  template <typename TFunction, typename ...Args>
  static mResult CallFunction(TFunction function, Args&&...args)
  {
    return function(std::forward<Args>(args)...);
  }
};

template<class TFunction, class Args, size_t ...TCountArgs>
void _mThread_ThreadInternal_CallFunctionUnpack(mThread *pThread, TFunction function, Args params, _mThread_ThreadInternalSequence<TCountArgs...>)
{
  pThread->result = _mThread_ThreadInternal_Decay<std::decay<TFunction>::type>::CallFunction(function, std::get<TCountArgs>(params) ...);
}

template<class TFunction, class Args>
void _mThread_ThreadInternalFunc(mThread *pThread, TFunction *pFunction, Args args)
{
  mASSERT(pThread != nullptr && pFunction != nullptr, "pThread and pFunction cannot be nullptr.");

  pThread->threadState = mT_TS_Running;

  _mThread_ThreadInternal_CallFunctionUnpack(pThread, *pFunction, args, typename _mThread_ThreadInternalIntegerSequenceGenerator<std::tuple_size<Args>::value>::type());

  pThread->threadState = mT_TS_Stopped;
}

// Creates and starts a thread.
template<class TFunction, class... Args>
mFUNCTION(mThread_Create, OUT mThread **ppThread, IN OPTIONAL mAllocator *pAllocator, TFunction *pFunction, Args&&... args)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppThread == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION_ON_ERROR(ppThread, mSetToNullptr);
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppThread));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppThread, 1));

  (*ppThread)->pAllocator = pAllocator;
  (*ppThread)->result = mR_Success;

  new (&(*ppThread)->threadState) std::atomic<mThread_ThreadState>(mT_TS_NotStarted);

  auto tupleRef = std::make_tuple(std::forward<Args>(args)...);
  new (&(*ppThread)->handle) std::thread (&_mThread_ThreadInternalFunc<TFunction, decltype(tupleRef)>, *ppThread, pFunction, tupleRef);
  
  mRETURN_SUCCESS();
}

mFUNCTION(mThread_Destroy, IN_OUT mThread **ppThread);
mFUNCTION(mThread_Join, IN mThread *pThread);
mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT std::thread::native_handle_type *pNativeHandle);

mFUNCTION(mThread_GetMaximumConcurrentThreads, OUT size_t *pMaximumConcurrentThreadCount);

#endif // mThread_h__
