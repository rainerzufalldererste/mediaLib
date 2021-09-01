#include "mPool.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "QbCK9wfNPslwGAxp4/ggb+L2RoE/pt3ikmEf0eISDuQuxOe84S8IuhG1E4466Pp6LT7dbEdLleYcK/hx"
#endif

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
  if (pool->count == pool->size * mBITS_OF(*pool->pIndexes))
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

    for (size_t indexOffset = 0; indexOffset < mBITS_OF(flags); ++indexOffset)
    {
      if ((flags & 1) == 0)
      {
        *pIndex = indexOffset + i * mBITS_OF(flags);
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

          new (pInItem) T(*pItem);
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
mFUNCTION(mPool_Add, mPtr<mPool<T>> &pool, IN T &&item, OUT size_t *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || pIndex == nullptr, mR_ArgumentNull);

  // Grow if necessary
  if (pool->count == pool->size * mBITS_OF(*pool->pIndexes))
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

    for (size_t indexOffset = 0; indexOffset < mBITS_OF(flags); ++indexOffset)
    {
      if ((flags & 1) == 0)
      {
        *pIndex = indexOffset + i * mBITS_OF(flags);
        size_t dataCount = 0;
        mERROR_CHECK(mChunkedArray_GetCount(pool->data, &dataCount));

        if (*pIndex == dataCount)
        {
          mERROR_CHECK(mChunkedArray_PushBack(pool->data, std::forward<T>(item)));
        }
        else
        {
          T *pInItem;
          mERROR_CHECK(mChunkedArray_PointerAt(pool->data, *pIndex, &pInItem));

          new (pInItem) T(std::move(item));
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
  mERROR_IF(index >= pool->size * mBITS_OF(pool->pIndexes[0]), mR_IndexOutOfBounds);

  const size_t lutIndex = index / mBITS_OF(pool->pIndexes[0]);
  const size_t lutSubIndex = index - lutIndex * mBITS_OF(pool->pIndexes[0]);

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
mFUNCTION(mPool_PeekAt, const mPtr<mPool<T>> &pool, const size_t index, OUT const T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pItem == nullptr, mR_ArgumentNull);

  const T *pOutItem = nullptr;
  mERROR_CHECK(mPool_PointerAt(pool, index, &pOutItem));

  *pItem = *pOutItem;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_PointerAt, mPtr<mPool<T>> &pool, const size_t index, OUT T **ppItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || ppItem == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= pool->size * mBITS_OF(pool->pIndexes[0]), mR_IndexOutOfBounds);

  const size_t lutIndex = index / mBITS_OF(pool->pIndexes[0]);
  const size_t lutSubIndex = index - lutIndex * mBITS_OF(pool->pIndexes[0]);

  mERROR_IF((pool->pIndexes[lutIndex] & ((size_t)1 << lutSubIndex)) == 0, mR_IndexOutOfBounds);

  mERROR_CHECK(mChunkedArray_PointerAt(pool->data, index, ppItem));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_PointerAt, const mPtr<mPool<T>> &pool, const size_t index, OUT const T **ppItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || ppItem == nullptr, mR_ArgumentNull);
  mERROR_IF(index >= pool->size * mBITS_OF(pool->pIndexes[0]), mR_IndexOutOfBounds);

  const size_t lutIndex = index / mBITS_OF(pool->pIndexes[0]);
  const size_t lutSubIndex = index - lutIndex * mBITS_OF(pool->pIndexes[0]);

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

    for (size_t j = 0; j < mBITS_OF(pool->pIndexes[0]); ++j)
    {
      if (pool->pIndexes[i] & flag)
      {
        T *pItem = nullptr;
        mERROR_CHECK(mChunkedArray_PointerAt(pool->data, index, &pItem));
        const mResult result = function(pItem, index);

        if (mFAILED(result))
        {
          if (result == mR_Break)
            break;
          else
            mRETURN_RESULT(result);
        }
      }

      flag <<= 1;
      ++index;
    }
  }

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_ContainsIndex, mPtr<mPool<T>> &pool, const size_t index, OUT bool *pContained)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr || pContained == nullptr, mR_ArgumentNull);
  
  if (index >= pool->size * mBITS_OF(pool->pIndexes[0]))
  {
    *pContained = false;
    mRETURN_SUCCESS(); 
  }

  const size_t lutIndex = index / mBITS_OF(pool->pIndexes[0]);
  const size_t lutSubIndex = index - lutIndex * mBITS_OF(pool->pIndexes[0]);

  if ((pool->pIndexes[lutIndex] & ((size_t)1 << lutSubIndex)) == 0)
  {
    *pContained = false;
    mRETURN_SUCCESS();
  }

  *pContained = true;

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mPool_Clear, mPtr<mPool<T>> &pool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pool == nullptr, mR_ArgumentNull);

  for (size_t i = 0; i < pool->size; i++)
    pool->pIndexes[i] = 0;

  pool->count = 0;

  mERROR_CHECK(mChunkedArray_Clear(pool->data));

  mRETURN_SUCCESS();
}

template <typename T, typename equals_func /* = mEqualsValue<T> */, typename element_valid_func /* = mTrue */>
bool mPool_Equals(const mPtr<mPool<T>> &a, const mPtr<mPool<T>> &b)
{
  if (a == b)
    return true;

  if ((a == nullptr) ^ (b == nullptr))
    return false;

  auto itA = a->Iterate();
  auto startA = itA.begin();
  auto endA = itA.end();

  auto itB = b->Iterate();
  auto startB = itB.begin();
  auto endB = itB.end();

  while (true)
  {
    if (!(startA != endA)) // no more values in a.
    {
      // if there's a value in `b` thats active: return false.
      while (startB != endB)
      {
        auto _b = *startB;

        if ((bool)element_valid_func()(_b.pData))
          return false;

        ++startB;
      }

      break;
    }
    else if (!(startB != endB)) // no more values in b, but values in a.
    {
      // if there's a value in `a` thats active: return false.
      do // do-while-loop, because we've already checked if (startA != endA) and an iterator might rely on that function only being called once.
      {
        auto _a = *startA;

        if ((bool)element_valid_func()(_a.pData))
          return false;

        ++startA;
      } while (startA != endA);

      break;
    }

    auto _a = *startA;
    bool end = false;

    while (!(bool)element_valid_func()(_a.pData))
    {
      ++startA;

      // if we've reached the end.
      if (!(startA != endA))
      {
        // if there's a value in `b` thats active: return false.
        while (startB != endB)
        {
          auto __b = *startB;

          if ((bool)element_valid_func()(__b.pData))
            return false;

          ++startB;
        }

        end = true;
        break;
      }

      _a = *startA;
    }

    if (end)
      break;

    auto _b = *startB;

    while (!(bool)element_valid_func()(_b.pData))
    {
      ++startB;

      // if we've reached the end.
      if (!(startB != endB))
        return false; // `a` is not at the end and valid.

      _b = *startB;
    }

    if (_a.index != _b.index || !(bool)equals_func()(_a.pData, _b.pData))
      return false;

    ++startA;
    ++startB;
  }

  return true;
}

template<typename T, typename T2, typename comparison>
inline bool mPool_ContainsValue(const mPtr<mPool<T>> &pool, const T2 &value, OUT OPTIONAL size_t *pIndex, comparison _comparison)
{
  if (pool == nullptr)
    return false;

  for (const auto &_item : queue->Iterate())
  {
    if (_comparison((*_item), value))
    {
      if (pIndex != nullptr)
        *pIndex = _item.index;

      return true;
    }
  }

  return false;
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

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mPoolIterator<T>::mPoolIterator(mPool<T> *pPool) :
  index(0),
  globalIndex(0),
  blockIndex(0),
  flag(1),
  pData(nullptr),
  pPool(pPool)
{ }

template<typename T>
inline typename mPoolIterator<T>::IteratorValue mPoolIterator<T>::operator*()
{
  return mPoolIterator<T>::IteratorValue(pData, globalIndex - 1);
}

template<typename T>
inline const typename mPoolIterator<T>::IteratorValue mPoolIterator<T>::operator*() const
{
  return mPoolIterator<T>::IteratorValue(pData, globalIndex - 1);
}

template<typename T>
inline bool mPoolIterator<T>::operator != (const typename mPoolIterator<T> &)
{
  for (blockIndex; blockIndex < pPool->size; ++blockIndex)
  {
    for (; index < mBITS_OF(pPool->pIndexes[0]); ++index)
    {
      if (pPool->pIndexes[blockIndex] & flag)
      {
        const mResult result = mChunkedArray_PointerAt(pPool->data, globalIndex, &pData);

        if (mFAILED(result))
        {
          mFAIL_DEBUG(mFormat("mChunkedArray_PointerAt failed in mPoolIterator::operator != with errorcode 0x", mFUInt<mFHex>(result), "."));
          return false;
        }

        flag <<= 1;
        ++globalIndex;
        ++index;

        return true;
      }

      flag <<= 1;
      ++globalIndex;
    }

    index = 0;
    flag = 1;
  }

  return false;
}

template<typename T>
inline typename mPoolIterator<T>& mPoolIterator<T>::operator++()
{
  return *this;
}

template<typename T>
inline mPoolIterator<T>::IteratorValue::IteratorValue(T *pData, const size_t index) :
  pData(pData),
  index(index)
{ }

template<typename T>
inline T& mPoolIterator<T>::IteratorValue::operator*()
{
  return *pData;
}

template<typename T>
inline const T& mPoolIterator<T>::IteratorValue::operator*() const
{
  return *pData;
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mConstPoolIterator<T>::mConstPoolIterator(const mPool<T> *pPool) :
  index(0),
  globalIndex(0),
  blockIndex(0),
  flag(1),
  pData(nullptr),
  pPool(pPool)
{ }

template<typename T>
inline const typename mConstPoolIterator<T>::IteratorValue mConstPoolIterator<T>::operator*() const
{
  return mConstPoolIterator<T>::IteratorValue(pData, globalIndex - 1);
}

template<typename T>
inline bool mConstPoolIterator<T>::operator != (const typename mConstPoolIterator<T> &)
{
  for (blockIndex; blockIndex < pPool->size; ++blockIndex)
  {
    for (; index < mBITS_OF(pPool->pIndexes[0]); ++index)
    {
      if (pPool->pIndexes[blockIndex] & flag)
      {
        const mResult result = mChunkedArray_PointerAt(pPool->data, globalIndex, &pData);

        if (mFAILED(result))
        {
          mFAIL_DEBUG(mFormat("mChunkedArray_ConstPointerAt failed in mConstPoolIterator::operator != with errorcode 0x", mFUInt<mFHex>(result), "."));
          return false;
        }

        flag <<= 1;
        ++globalIndex;
        ++index;

        return true;
      }

      flag <<= 1;
      ++globalIndex;
    }

    index = 0;
    flag = 1;
  }

  return false;
}

template<typename T>
inline typename mConstPoolIterator<T>& mConstPoolIterator<T>::operator++()
{
  return *this;
}

template<typename T>
inline mConstPoolIterator<T>::IteratorValue::IteratorValue(const T *pData, const size_t index) :
  pData(pData),
  index(index)
{ }

template<typename T>
inline const T& mConstPoolIterator<T>::IteratorValue::operator*() const
{
  return *pData;
}
