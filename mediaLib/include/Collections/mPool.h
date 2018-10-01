#ifndef mPool_h__
#define mPool_h__

#include "mediaLib.h"
#include "mChunkedArray.h"

template <typename T>
struct mPool
{
  size_t count;
  size_t size;
  size_t allocatedSize;
  size_t *pIndexes;
  mPtr<mChunkedArray<T>> data;
  mAllocator *pAllocator;
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
