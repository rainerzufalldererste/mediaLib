#ifndef mRefPool_h__
#define mRefPool_h__

#include "mediaLib.h"
#include "mPool.h"
#include "mMutex.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "VDUnWw5Oa6IyfzA4A4LqgrxCxA+HrRsNKbVRqdHlmO1EtCUpzGIc7LVfyQuHBcLHO987drMzipRlUkT+"
#endif

template <typename T>
struct mRefPool_SharedPointerContainer_Internal
{
  size_t dataIndex;
  mPtr<T> ptr;
};

template <typename T>
struct mRefPoolIterator
{
  mRefPoolIterator(mPool<mRefPool_SharedPointerContainer_Internal<T>> *pPool);

  struct IteratorValue
  {
    IteratorValue(mPtr<T> &data, const size_t index);

    T& operator *();
    const T& operator *() const;

    mPtr<T> data;
    size_t index;
  };

  typename mRefPoolIterator<T>::IteratorValue operator *();
  const typename mRefPoolIterator<T>::IteratorValue operator *() const;
  bool operator != (const typename mRefPoolIterator<T> &);
  typename mRefPoolIterator<T>& operator++();

private:
  size_t index, blockIndex, flag, globalIndex;
  mRefPool_SharedPointerContainer_Internal<T> *pData;
  mPool<mRefPool_SharedPointerContainer_Internal<T>> *pPool;
};

template <typename T>
struct mRefPoolConstIterator
{
  mRefPoolConstIterator(const mPool<mRefPool_SharedPointerContainer_Internal<T>> *pPool);

  struct IteratorValue
  {
    IteratorValue(const mPtr<T> &data, const size_t index);

    const T& operator *() const;

    const mPtr<T> data;
    size_t index;
  };

  const typename mRefPoolConstIterator<T>::IteratorValue operator *() const;
  bool operator != (const typename mRefPoolConstIterator<T> &);
  typename mRefPoolConstIterator<T>& operator++();

private:
  size_t index, blockIndex, flag, globalIndex;
  const mRefPool_SharedPointerContainer_Internal<T> *pData;
  const mPool<mRefPool_SharedPointerContainer_Internal<T>> *pPool;
};

template <typename T>
struct mRefPool
{
  struct refPoolPtrData
  {
    T element;
    typename mSharedPointer<T>::PointerParams ptrParams;
    size_t index;
  };

  mPtr<mPool<refPoolPtrData>> data;
  mPtr<mPool<mRefPool_SharedPointerContainer_Internal<T>>> ptrs;
  mRecursiveMutex *pMutex;
  mAllocator *pAllocator;
  bool keepForever;

  struct
  {
    typename mRefPoolIterator<T> begin() { return mRefPoolIterator<T>(pPool); };
    typename mRefPoolIterator<T> end() { return mRefPoolIterator<T>(pPool); };

    mPool<mRefPool_SharedPointerContainer_Internal<T>> *pPool;
  } Iterate() { return {this->ptrs.GetPointer()}; };

  struct
  {
    typename mRefPoolConstIterator<T> begin() { return mRefPoolConstIterator<T>(pPool); };
    typename mRefPoolConstIterator<T> end() { return mRefPoolConstIterator<T>(pPool); };

    const mPool<mRefPool_SharedPointerContainer_Internal<T>> *pPool;
  } Iterate() const { return {this->ptrs.GetPointer()}; };
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

template <typename T>
mFUNCTION(mRefPool_ContainsIndex, mPtr<mRefPool<T>> &refPool, const size_t index, OUT bool *pIsContained);

// would be handled by cpp but still nicer if explicitly defined.
template <typename T>
mFUNCTION(mDestruct, IN struct mRefPool_SharedPointerContainer_Internal<T> *pData);

template <typename T, typename equals_func, typename element_valid_func = mTrue>
bool mRefPool_Equals(const mPtr<mRefPool<T>> &a, const mPtr<mRefPool<T>> &b);

#include "mRefPool.inl"

#endif // mRefPool_h__
