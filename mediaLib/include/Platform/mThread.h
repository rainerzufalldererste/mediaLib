#ifndef mThread_h__
#define mThread_h__

#include "mediaLib.h"
#include <thread>
#include <atomic>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "knhRzjiI9qHoYHJxO3TK+GhMsCUBkyk7oLCcI0Te/luC6f1iMPArIT02KLkiMbNemRC0ZVJZYfCRSDOm"
#endif

enum mThread_ThreadState
{
  mT_TS_NotStarted = 0,
  mT_TS_Running,
  mT_TS_Stopped,
};

struct mThread
{
  mAllocator *pAllocator;
#if defined(mPLATFORM_WINDOWS)
  HANDLE handle;
  std::function<void(mThread *)> threadFunc;
#else
  std::thread handle;
#endif
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

template<typename TFunctionDecay>
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

template<typename TFunction, typename Args, size_t ...TCountArgs>
void _mThread_ThreadInternal_CallFunctionUnpack(mThread *pThread, TFunction function, Args params, _mThread_ThreadInternalSequence<TCountArgs...>)
{
  mUnused(params); // Ignore warnings in case the function does not take any arguments.
  pThread->result = _mThread_ThreadInternal_Decay<decltype(function(std::get<TCountArgs>(params)...))>::CallFunction(function, std::get<TCountArgs>(params) ...);
}

template<typename TFunction, typename Args>
void _mThread_ThreadInternalFunc(mThread *pThread, TFunction function, Args args)
{
  mASSERT(pThread != nullptr || function == nullptr, "pThread cannot be nullptr.");

  pThread->threadState = mT_TS_Running;

  _mThread_ThreadInternal_CallFunctionUnpack(pThread, function, args, typename _mThread_ThreadInternalIntegerSequenceGenerator<std::tuple_size<Args>::value>::type());

  pThread->threadState = mT_TS_Stopped;
}

#if defined(mPLATFORM_WINDOWS)
mFUNCTION(_mThread_CreateHandleInternal, IN mThread *pThread);
#endif

// Creates and starts a thread.
template<typename TFunction, typename... Args>
mFUNCTION(mThread_Create, OUT mThread **ppThread, IN OPTIONAL mAllocator *pAllocator, TFunction function, Args&&... args)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppThread == nullptr || function == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(ppThread, mSetToNullptr);
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppThread));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppThread, 1));

  (*ppThread)->pAllocator = pAllocator;
  (*ppThread)->result = mR_Success;

  new (&(*ppThread)->threadState) std::atomic<mThread_ThreadState>(mT_TS_NotStarted);

  auto tupleRef = std::make_tuple(std::forward<Args>(args)...);

#if defined(mPLATFORM_WINDOWS)
  new (&(*ppThread)->threadFunc) std::function<void (mThread *)>([=](mThread *pThread) { _mThread_ThreadInternalFunc<TFunction, decltype(tupleRef)>(pThread, function, tupleRef); });

  mERROR_CHECK(_mThread_CreateHandleInternal(*ppThread));
#else
  new (&(*ppThread)->handle) std::thread (&_mThread_ThreadInternalFunc<TFunction, decltype(tupleRef)>, *ppThread, function, tupleRef);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mThread_Destroy, IN_OUT mThread **ppThread);
mFUNCTION(mThread_Join, IN mThread *pThread);

#if defined(mPLATFORM_WINDOWS)
mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT HANDLE *pNativeHandle);
#else
mFUNCTION(mThread_GetNativeHandle, IN mThread *pThread, OUT std::thread::native_handle_type *pNativeHandle);
#endif

mFUNCTION(mThread_GetMaximumConcurrentThreads, OUT size_t *pMaximumConcurrentThreadCount);

enum mThread_Priority
{
  mT_P_Low,
  mT_P_Normal,
  mT_P_High,
  mT_P_Realtime,
};

mFUNCTION(mThread_SetThreadPriority, IN mThread *pThread, const mThread_Priority priority);
mFUNCTION(mThread_SetCurrentThreadPriority, const mThread_Priority priority);

#endif // mThread_h__
