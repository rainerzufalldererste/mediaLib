#ifndef mPool_h__
#define mPool_h__

#include "mediaLib.h"
#include "mChunkedArray.h"

template <typename T>
struct mPool;

template <typename T>
struct mPoolIterator
{
  mPoolIterator(mPool<T> *pPool);

  struct IteratorValue
  {
    IteratorValue(T *pData, const size_t index);

    T& operator *();
    const T& operator *() const;

    T *pData;
    size_t index;
  };

  typename mPoolIterator<T>::IteratorValue operator *();
  const typename mPoolIterator<T>::IteratorValue operator *() const;
  bool operator != (const typename mPoolIterator<T> &);
  typename mPoolIterator<T>& operator++();

private:
  size_t index, blockIndex, flag, globalIndex;
  T *pData;
  mPool<T> *pPool;
};

template <typename T>
struct mPool
{
  size_t count;
  size_t size;
  size_t allocatedSize;
  size_t *pIndexes;
  mPtr<mChunkedArray<T>> data;
  mAllocator *pAllocator;

  struct
  {
    typename mPoolIterator<T> begin() { return mPoolIterator<T>(pPool); };
    typename mPoolIterator<T> end() { return mPoolIterator<T>(pPool); };

    mPool<T> *pPool;
  } Iterate() { return {this}; };
};

template <typename T>
mFUNCTION(mPool_Create, OUT mPtr<mPool<T>> *pPool, IN mAllocator *pAllocator);

template <typename T>
mFUNCTION(mPool_Destroy, IN_OUT mPtr<mPool<T>> *pPool);

template <typename T>
mFUNCTION(mPool_Add, mPtr<mPool<T>> &pool, IN T *pItem, OUT size_t *pIndex);

template <typename T>
mFUNCTION(mPool_Add, mPtr<mPool<T>> &pool, IN T &&item, OUT size_t *pIndex);

template <typename T>
mFUNCTION(mPool_RemoveAt, mPtr<mPool<T>> &pool, const size_t index, OUT T *pItem);

template <typename T>
mFUNCTION(mPool_GetCount, mPtr<mPool<T>> &pool, OUT size_t *pCount);

template <typename T>
mFUNCTION(mPool_PeekAt, mPtr<mPool<T>> &pool, const size_t index, OUT T *pItem);

template <typename T>
mFUNCTION(mPool_PointerAt, mPtr<mPool<T>> &pool, const size_t index, OUT T **ppItem);

template <typename T>
mFUNCTION(mPool_ForEach, mPtr<mPool<T>> &pool, const std::function<mResult (T *, size_t)> &function);

#include "mPool.inl"

#endif // mPool_h__
