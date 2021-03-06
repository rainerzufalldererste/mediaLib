#include "mPool.h"
// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

template <typename T>
mFUNCTION(mPool_Destroy_Internal, IN mPool<T> *pPool);

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mPool_Create, OUT mPtr<mPool<T>> *pPool, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pPool, pAllocator, (std::function<void(mPool<T> *)>) [](mPool<T> *pData) { mPool_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mChunkedArray_Create(&(*pPool)->data, pAllocator));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_Destroy, IN_OUT mPtr<mPool<T>> *pPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pPool));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_Add, mPtr<mPool<T>> &pool, IN T *pItem, OUT size_t *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || pItem == nullptr || pIndex == nullptr, mR_ArgumentNull);

  // Grow if necessary
  if (pool->count == pool->size * mBYTES_OF(*pool->pIndexes))
  {
    if (pool->allocatedSize == pool->size)
    {
      const size_t newAllocatedSize = pool->allocatedSize * 2 + 1;
      mERROR_CHECK(mAllocator_Reallocate(pool->pAllocator, &pool->pIndexes, newAllocatedSize));
      pool->allocatedSize = newAllocatedSize;
    }

    pool->pIndexes[pool->size] = 0;
    ++pool->size;
  }

  for (size_t i = 0; i < pool->size; ++i)
  {
    std::remove_reference<decltype(*pool->pIndexes)>::type flags = pool->pIndexes[i];

    if (flags == (size_t)-1)
      continue;

    for (size_t indexOffset = 0; indexOffset < mBYTES_OF(flags); ++indexOffset)
    {
      if ((flags & 1) == 0)
      {
        *pIndex = indexOffset + i * mBYTES_OF(flags);
        size_t dataCount = 0;
        mERROR_CHECK(mChunkedArray_GetCount(pool->data, &dataCount));
        
        if (*pIndex == dataCount)
        {
          mERROR_CHECK(mChunkedArray_PushBack(pool->data, pItem));
        }
        else
        {
          T *pInItem;
          mERROR_CHECK(mChunkedArray_PointerAt(pool->data, *pIndex, &pInItem));

          new (pInItem) T (*pItem);
        }

        pool->pIndexes[i] |= ((size_t)1 << indexOffset);
        ++pool->count;

        goto break_all_loops;
      }

      flags >>= 1;
    }
  }

break_all_loops:

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_RemoveAt, mPtr<mPool<T>> &pool, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= pool->size * mBYTES_OF(pool->pIndexes[0]), mR_IndexOutOfBounds);

  const size_t lutIndex = index / mBYTES_OF(pool->pIndexes[0]);
  const size_t lutSubIndex = index - lutIndex * mBYTES_OF(pool->pIndexes[0]);

  mERROR_IF((pool->pIndexes[lutIndex] & ((size_t)1 << lutSubIndex)) == 0, mR_IndexOutOfBounds);

  T *pOutItem = nullptr;
  mERROR_CHECK(mPool_PointerAt(pool, index, &pOutItem));

  *pItem = std::move(*pOutItem);
  mERROR_CHECK(mMemset(pOutItem, 1));
  --pool->count;
  pool->pIndexes[lutIndex] &= ~((size_t)1 << lutSubIndex);

  if (lutIndex == pool->size - 1)
  {
    for (int64_t lutFilledIndex = pool->size - 1; lutFilledIndex >= 0; --lutFilledIndex)
    {
      if (pool->pIndexes[lutFilledIndex] == 0)
        --pool->size;
      else
        break;
    }
  }

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_GetCount, mPtr<mPool<T>> &pool, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = pool->count;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_PeekAt, mPtr<mPool<T>> &pool, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pItem == nullptr, mR_ArgumentNull);

  T *pOutItem = nullptr;
  mERROR_CHECK(mPool_PointerAt(pool, index, &pOutItem));

  *pItem = *pOutItem;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_PointerAt, mPtr<mPool<T>> &pool, const size_t index, OUT T **ppItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || ppItem == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= pool->size * mBYTES_OF(pool->pIndexes[0]), mR_IndexOutOfBounds);

  const size_t lutIndex = index / mBYTES_OF(pool->pIndexes[0]);
  const size_t lutSubIndex = index - lutIndex * mBYTES_OF(pool->pIndexes[0]);

  mERROR_IF((pool->pIndexes[lutIndex] & ((size_t)1 << lutSubIndex)) == 0, mR_IndexOutOfBounds);

  mERROR_CHECK(mChunkedArray_PointerAt(pool->data, index, ppItem));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mPool_ForEach, mPtr<mPool<T>> &pool, const std::function<mResult(T *, size_t)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || function == nullptr, mR_ArgumentNull);

  size_t index = 0;
  
  for (size_t i = 0; i < pool->size; ++i)
  {
    size_t flag = 1;

    for (size_t j = 0; j < mBYTES_OF(pool->pIndexes[0]); ++j)
    {
      if (pool->pIndexes[i] & flag)
      {
        T *pItem = nullptr;
        mERROR_CHECK(mChunkedArray_PointerAt(pool->data, index, &pItem));
        mERROR_CHECK(function(pItem, index));
      }

      flag <<= 1;
      ++index;
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mPool_Destroy_Internal, IN mPool<T> *pPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mChunkedArray_Destroy(&pPool->data));

  if (pPool->allocatedSize > 0)
    mERROR_CHECK(mAllocator_FreePtr(pPool->pAllocator, &pPool->pIndexes));

  mRETURN_SUCCESS();
}
