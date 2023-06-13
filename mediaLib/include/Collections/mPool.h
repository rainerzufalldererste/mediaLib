#ifndef mPool_h__
#define mPool_h__

#include "mediaLib.h"
#include "mChunkedArray.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "zvER/mmueuwZoijapyOovjPwpKJTEr+rQRMlTXypmnBitB+ZFxvwPFwNy2B8gFFzvpzcfBDKSHE0wFfK"
#endif

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
struct mConstPoolIterator
{
  mConstPoolIterator(const mPool<T> *pPool);

  struct IteratorValue
  {
    IteratorValue(const T *pData, const size_t index);

    const T& operator *() const;

    const T *pData;
    size_t index;
  };

  const typename mConstPoolIterator<T>::IteratorValue operator *() const;
  bool operator != (const typename mConstPoolIterator<T> &);
  typename mConstPoolIterator<T>& operator++();

private:
  size_t index, blockIndex, flag, globalIndex;
  const T *pData;
  const mPool<T> *pPool;
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
  } Iterate() { return{ this }; };

  struct
  {
    typename mConstPoolIterator<T> begin() { return mConstPoolIterator<T>(pPool); };
    typename mConstPoolIterator<T> end() { return mConstPoolIterator<T>(pPool); };

    const mPool<T> *pPool;
  } Iterate() const { return {this}; };
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
mFUNCTION(mPool_GetCount, const mPtr<mPool<T>> &pool, OUT size_t *pCount);

template <typename T>
mFUNCTION(mPool_PeekAt, mPtr<mPool<T>> &pool, const size_t index, OUT T *pItem);

template <typename T>
mFUNCTION(mPool_PeekAt, const mPtr<mPool<T>> &pool, const size_t index, OUT const T *pItem);

template <typename T>
mFUNCTION(mPool_PointerAt, mPtr<mPool<T>> &pool, const size_t index, OUT T **ppItem);

template <typename T>
mFUNCTION(mPool_PointerAt, const mPtr<mPool<T>> &pool, const size_t index, OUT T * const *ppItem);

template <typename T>
mFUNCTION(mPool_ForEach, mPtr<mPool<T>> &pool, const std::function<mResult (T *, size_t)> &function);

template <typename T>
mFUNCTION(mPool_ContainsIndex, const mPtr<mPool<T>> &pool, const size_t index, OUT bool *pContained);

template <typename T>
mFUNCTION(mPool_Clear, mPtr<mPool<T>> &pool);

template <typename T, typename equals_func = mEqualsValue<T>, typename element_valid_func = mTrue>
bool mPool_Equals(const mPtr<mPool<T>> &a, const mPtr<mPool<T>> &b);

template <typename T, typename T2, typename comparison = mEquals<T, T2>>
bool mPool_ContainsValue(const mPtr<mPool<T>> &pool, const T2 &value, OUT OPTIONAL size_t *pIndex = nullptr, comparison _comparison = comparison());

#include "mPool.inl"

#endif // mPool_h__
