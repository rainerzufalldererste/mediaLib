#ifndef mRefPool_h__
#define mRefPool_h__

#include "mediaLib.h"
#include "mPool.h"
#include "mMutex.h"

template <typename T>
struct mRefPool
{
  struct refPoolPtrData
  {
    T element;
    typename mSharedPointer<T>::PointerParams ptrParams;
    size_t index;
  };

  struct refPoolPtr
  {
    size_t dataIndex;
    mPtr<T> ptr;
  };

  mPtr<mPool<refPoolPtrData>> data;
  mPtr<mPool<refPoolPtr>> ptrs;
  mRecursiveMutex *pMutex;
  mAllocator *pAllocator;
  bool keepForever;
};

template <typename T>
mFUNCTION(mRefPool_Create, OUT mPtr<mRefPool<T>> *pRefPool, IN mAllocator *pAllocator, const bool keepEntriesForever = false);

template <typename T>
mFUNCTION(mRefPool_Destroy, IN_OUT mPtr<mRefPool<T>> *pRefPool);

template <typename T>
mFUNCTION(mRefPool_Add, mPtr<mRefPool<T>> &refPool, IN T *pItem, OUT mPtr<T> *pIndex);

template <typename T>
mFUNCTION(mRefPool_AddEmpty, mPtr<mRefPool<T>> &refPool, OUT mPtr<T> *pIndex);

template <typename T>
mFUNCTION(mRefPool_Crush, mPtr<mRefPool<T>> &refPool);

template <typename T>
mFUNCTION(mRefPool_ForEach, mPtr<mRefPool<T>> &refPool, const std::function<mResult(mPtr<T> &)> &function);

// This will only remove the pools own reference if `keepEntriesForever` is true (in `mRefPool_Create`) and the pool is the only holder of a reference.
template <typename T>
mFUNCTION(mRefPool_KeepIfTrue, mPtr<mRefPool<T>> &refPool, const std::function<mResult(mPtr<T> &, OUT bool *)> &function);

template <typename T>
mFUNCTION(mRefPool_PeekAt, mPtr<mRefPool<T>> &refPool, const size_t index, OUT mPtr<T> *pIndex);

template <typename T>
mFUNCTION(mRefPool_GetCount, mPtr<mRefPool<T>> &refPool, OUT size_t *pCount);

template <typename T>
mFUNCTION(mRefPool_RemoveOwnReference, mPtr<mRefPool<T>> &refPool);

template <typename T>
mFUNCTION(mRefPool_GetPointerIndex, const mPtr<T> &ptr, size_t *pIndex);

// would be handled by cpp but still nicer if explicitly defined.
template <typename T>
mFUNCTION(mDestruct, IN struct mRefPool<T>::refPoolPtr *pData);

#include "mRefPool.inl"

#endif // mRefPool_h__
