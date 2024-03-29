#ifndef mChunkedArray_h__
#define mChunkedArray_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "38LlZaK67nGjK8C3zG2FSFwK33Bkf27LvLeKNe4NdP93egxNwyROpaEyw4DxtjcizpNMXQ0A8sG0WN7C"
#endif

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
mFUNCTION(mChunkedArray_PointerAt, const mPtr<mChunkedArray<T>> &chunkedArray, const size_t index, OUT T * const *ppItem);

template <typename T>
mFUNCTION(mChunkedArray_Clear, mPtr<mChunkedArray<T>> &chunkedArray);

template <typename T>
mFUNCTION(mChunkedArray_SetDestructionFunction, mPtr<mChunkedArray<T>> &chunkedArray, const std::function<mResult(T *)> &destructionFunction);

#include "mChunkedArray.inl"

#endif // mChunkedArray_h__
