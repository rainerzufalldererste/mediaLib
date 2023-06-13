#ifndef mChunkedArray_h__
#define mChunkedArray_h__

#include "mediaLib.h"

template <typename T>
struct mChunkedArray
{
  struct mChunkedArrayBlock
  {
    T *pData;
    size_t blockSize;
    size_t blockStartIndex;
    size_t blockEndIndex;
  };

  size_t blockSize;
  size_t blockCount;
  size_t blockCapacity;
  size_t itemCount;
  size_t itemCapacity;
  mChunkedArrayBlock *pBlocks;
  mAllocator *pAllocator;
  std::function<mResult(T *)> destructionFunction;
};

template <typename T>
mFUNCTION(mChunkedArray_Create, OUT mPtr<mChunkedArray<T>> *pChunkedArray, IN mAllocator *pAllocator, const size_t blockSize = 64);

template <typename T>
mFUNCTION(mChunkedArray_Destroy, IN_OUT mPtr<mChunkedArray<T>> *pChunkedArray);

template <typename T>
mFUNCTION(mChunkedArray_Push, mPtr<mChunkedArray<T>> &chunkedArray, IN T *pItem, OUT OPTIONAL size_t *pIndex);

template <typename T>
mFUNCTION(mChunkedArray_PushBack, mPtr<mChunkedArray<T>> &chunkedArray, IN T *pItem);

template <typename T>
mFUNCTION(mChunkedArray_PushBack, mPtr<mChunkedArray<T>> &chunkedArray, IN T &&item);

template <typename T>
mFUNCTION(mChunkedArray_GetCount, mPtr<mChunkedArray<T>> &chunkedArray, OUT size_t *pCount);

template <typename T>
mFUNCTION(mChunkedArray_PeekAt, mPtr<mChunkedArray<T>> &chunkedArray, const size_t index, OUT T *pItem);

template <typename T>
mFUNCTION(mChunkedArray_PopAt, mPtr<mChunkedArray<T>> &chunkedArray, const size_t index, OUT T *pItem);

template <typename T>
mFUNCTION(mChunkedArray_PointerAt, mPtr<mChunkedArray<T>> &chunkedArray, const size_t index, OUT T **ppItem);

template <typename T>
mFUNCTION(mChunkedArray_SetDestructionFunction, mPtr<mChunkedArray<T>> &chunkedArray, const std::function<mResult(T *)> &destructionFunction);

#include "mChunkedArray.inl"

#endif // mChunkedArray_h__
