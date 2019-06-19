#ifndef mQueue_h__
#define mQueue_h__

#include "mediaLib.h"

template <typename T>
struct mQueueIterator
{
  ptrdiff_t direction;
  T *pCurrent;
  T *pLineEnd;
  T *pLineStart;
  T *pLastPos;
  size_t count;
  size_t index;

  mQueueIterator(const ptrdiff_t direction, T *pFirst, T *pLineEnd, T *pLineStart, const size_t count);
  T& operator *();
  const T& operator *() const;
  bool operator != (const mQueueIterator<T> &iterator) const;
  mQueueIterator<T>& operator++();
};

template <typename T>
struct mConstQueueIterator : private mQueueIterator<T>
{
  mConstQueueIterator(const ptrdiff_t direction, const T *pFirst, const T *pLineEnd, const T *pLineStart, const size_t count);
  const T& operator *() const;
  bool operator != (const mConstQueueIterator<T> &iterator) const;
  mConstQueueIterator<T>& operator++();
};

template <typename T>
struct mQueue
{
  mAllocator *pAllocator;
  T *pData;
  size_t startIndex, size, count;

  mQueueIterator<T> begin()
  {
    return mQueueIterator<T>(1, pData + startIndex, pData + size - 1, pData, count);
  }

  mQueueIterator<T> end()
  {
    if (size > 0)
      return mQueueIterator<T>(-1, pData + ((startIndex + count - 1) % size), pData + size - 1, pData, count);
    else
      return begin();
  };

  mConstQueueIterator<T> begin() const
  {
    return mConstQueueIterator<T>(1, pData + startIndex, pData + size - 1, pData, count);
  }

  mConstQueueIterator<T> end() const
  {
    if (size > 0)
      return mConstQueueIterator<T>(-1, pData + ((startIndex + count - 1) % size), pData + size - 1, pData, count);
    else
      return begin();
  };

  struct mQueueReverseIterator
  {
    mQueue<T> *pQueue;

    mQueueIterator<T> begin() { return pQueue->end(); }
    mQueueIterator<T> end() { return pQueue->begin(); }
  } IterateReverse() { return{ this }; };

  struct mQueueConstReverseIterator
  {
    const mQueue<T> *pQueue;

    mConstQueueIterator<T> begin() const { return pQueue->end(); }
    mConstQueueIterator<T> end() const { return pQueue->begin(); }
  } IterateReverse() const { return{ this }; };

  struct mQueueIteratorWrapper
  {
    mQueue<T> *pQueue;

    mQueueIterator<T> begin() { return pQueue->begin(); }
    mQueueIterator<T> end() { return pQueue->end(); }

  } Iterate() { return{ this }; };

  struct mQueueConstIteratorWrapper
  {
    const mQueue<T> *pQueue;

    mConstQueueIterator<T> begin() const { return pQueue->begin(); }
    mConstQueueIterator<T> end() const { return pQueue->end(); }

  } Iterate() const { return{ this }; };

  inline T& operator[](const size_t index)
  {
    return pData[(index + startIndex) % size];
  }

  inline const T& operator[](const size_t index) const
  {
    return pData[(index + startIndex) % size];
  }
};

template <typename T>
mFUNCTION(mQueue_Create, OUT mPtr<mQueue<T>> *pQueue, IN OPTIONAL mAllocator *pAllocator);

template <typename T>
mFUNCTION(mQueue_Create, OUT mUniqueContainer<mQueue<T>> *pQueue, IN OPTIONAL mAllocator *pAllocator);

template <typename T>
mFUNCTION(mQueue_Destroy, IN_OUT mPtr<mQueue<T>> *pQueue);

template <typename T>
mFUNCTION(mQueue_PushBack, mPtr<mQueue<T>> &queue, IN T *pItem);

template <typename T>
mFUNCTION(mQueue_PushBack, mPtr<mQueue<T>> &queue, IN T item);

template <typename T>
mFUNCTION(mQueue_PushFront, mPtr<mQueue<T>> &queue, IN T *pItem);

template <typename T>
mFUNCTION(mQueue_PushFront, mPtr<mQueue<T>> &queue, IN T item);

template <typename T>
mFUNCTION(mQueue_PopFront, mPtr<mQueue<T>> &queue, OUT T *pItem);

template <typename T>
mFUNCTION(mQueue_PeekFront, mPtr<mQueue<T>> &queue, OUT T *pItem);

template <typename T>
mFUNCTION(mQueue_PeekBack, mPtr<mQueue<T>> &queue, OUT T *pItem);

template <typename T>
mFUNCTION(mQueue_PopBack, mPtr<mQueue<T>> &queue, OUT T *pItem);

template <typename T>
mFUNCTION(mQueue_PeekAt, mPtr<mQueue<T>> &queue, const size_t index, OUT T *pItem);

template <typename T>
mFUNCTION(mQueue_PopAt, mPtr<mQueue<T>> &queue, size_t index, OUT T *pItem);

// Handle with care. The retrieved pointer will just be valid until the size changes or an element is removed.
template <typename T>
mFUNCTION(mQueue_PointerAt, mPtr<mQueue<T>> &queue, const size_t index, OUT T **ppItem);

template <typename T>
mFUNCTION(mQueue_GetCount, const mPtr<mQueue<T>> &queue, OUT size_t *pCount);

template <typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mQueue_Reserve, mPtr<mQueue<T>> &queue, const size_t count);

template <typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mQueue_Reserve, mPtr<mQueue<T>> &queue, const size_t count);

template <typename T>
mFUNCTION(mQueue_Clear, mPtr<mQueue<T>> &queue);

template <typename T>
mFUNCTION(mQueue_CopyTo, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator);

template <typename T>
mFUNCTION(mQueue_OrderBy, mPtr<mQueue<T>> &queue, const std::function<mComparisonResult(const T &a, const T &b)> &orderByFunc);

template <typename T, typename TLessFunc = std::less<T>, typename TGreaterFunc = std::greater<T>>
mFUNCTION(mQueue_OrderBy, mPtr<mQueue<T>> &queue);

template <typename T>
mFUNCTION(mQueue_Select, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator, const std::function<bool(const T &a)> &selectFunc);

template <typename T, typename Func>
mFUNCTION(mQueue_Select, const mPtr<mQueue<T>> &queue, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator, Func func = Func());

template <typename T>
mFUNCTION(mQueue_Apply, mPtr<mQueue<T>> &queue, const std::function<mResult(T *pValue)> &func);

template <typename T, typename Func>
mFUNCTION(mQueue_Apply, const mPtr<mQueue<T>> &queue);

template <typename T>
mFUNCTION(mQueue_Min, const mPtr<mQueue<T>> &queue, const std::function<mComparisonResult(const T &a, const T &b)> &comparisonFunc, OUT T *pMin);

template <typename T, typename comparison = std::less<T>>
mFUNCTION(mQueue_Min, const mPtr<mQueue<T>> &queue, OUT T *pMin, comparison _comparison = comparison());

template <typename T>
mFUNCTION(mQueue_Max, const mPtr<mQueue<T>> &queue, const std::function<mComparisonResult(const T &a, const T &b)> &comparisonFunc, OUT T *pMax);

template <typename T, typename comparison = std::greater<T>>
mFUNCTION(mQueue_Max, const mPtr<mQueue<T>> &queue, OUT T *pMax, comparison _comparison = comparison());

template <typename T, typename T2>
mFUNCTION(mQueue_Contains, const mPtr<mQueue<T>> &queue, const T2 &value, const std::function<bool(const T &a, const T2 &b)> &equalityComparer, OUT bool *pContained, OUT OPTIONAL size_t *pIndex = nullptr);

template <typename T, typename T2, typename comparison = mEquals<T, T2>>
mFUNCTION(mQueue_Contains, const mPtr<mQueue<T>> &queue, const T2 &value, OUT bool *pContained, OUT OPTIONAL size_t *pIndex = nullptr, comparison _comparison = comparison());

template <typename T>
mFUNCTION(mQueue_RemoveDuplicates, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator, const std::function<bool(const T &a, const T &b)> &equalityComparer);

template <typename T, typename comparison = std::equal_to<T>>
mFUNCTION(mQueue_RemoveDuplicates, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator);

template <typename T>
mFUNCTION(mQueue_Any, const mPtr<mQueue<T>> &queue, OUT bool *pAny);

//////////////////////////////////////////////////////////////////////////

template<typename T>
mFUNCTION(mQueue_Destroy_Internal, IN mQueue<T> *pQueue);

template<typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mQueue_Grow_Internal, mPtr<mQueue<T>> &queue);

template<typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mQueue_Grow_Internal, mPtr<mQueue<T>> &queue);

template <typename T>
mFUNCTION(mQueue_PopAt_Internal, mPtr<mQueue<T>> &queue, const size_t index, OUT T *pItem);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mQueue_Create, OUT mPtr<mQueue<T>> *pQueue, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pQueue == nullptr, mR_ArgumentNull);

  if (*pQueue != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pQueue));
    *pQueue = nullptr;
  }

  mQueue<T> *pQueueRaw = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pQueueRaw));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pQueueRaw, 1));

  mDEFER_CALL_ON_ERROR(pQueue, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create(pQueue, pQueueRaw, (std::function<void(mQueue<T> *)>)[](mQueue<T> *pData) { mQueue_Destroy_Internal<T>(pData); }, pAllocator));
  pQueueRaw = nullptr;

  (*pQueue)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Create, OUT mUniqueContainer<mQueue<T>> *pQueue, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pQueue == nullptr, mR_ArgumentNull);

  mUniqueContainer<mQueue<T>>::ConstructWithCleanupFunction(pQueue, [](mQueue<T> *pData) { mQueue_Destroy_Internal<T>(pData); });

  (*pQueue)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Destroy, IN_OUT mPtr<mQueue<T>> *pQueue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pQueue == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pQueue));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PushBack, mPtr<mQueue<T>> &queue, IN T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);

  if (queue->pData == nullptr)
    mERROR_CHECK(mQueue_Reserve(queue, 1));
  else if (queue->size == queue->count)
    mERROR_CHECK(mQueue_Grow_Internal(queue));

  const size_t index = (queue->startIndex + queue->count) % queue->size;

  if (!std::is_trivially_copy_constructible<T>::value)
    new (&queue->pData[index]) T(*pItem);
  else
    mERROR_CHECK(mAllocator_Copy(queue->pAllocator, &queue->pData[index], pItem, 1));

  ++queue->count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PushBack, mPtr<mQueue<T>> &queue, IN T item)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PushBack(queue, &item));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PushFront, mPtr<mQueue<T>> &queue, IN T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);

  if (queue->pData == nullptr)
    mERROR_CHECK(mQueue_Reserve(queue, 1));
  else if (queue->size == queue->count)
    mERROR_CHECK(mQueue_Grow_Internal(queue));

  size_t index;

  if (queue->startIndex > 0)
    index = queue->startIndex - 1;
  else
    index = queue->size - 1;

  if (!std::is_trivially_copy_constructible<T>::value)
    new (&queue->pData[index]) T(*pItem);
  else
    mERROR_CHECK(mAllocator_Copy(queue->pAllocator, &queue->pData[index], pItem, 1));

  ++queue->count;
  queue->startIndex = index;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PushFront, mPtr<mQueue<T>> &queue, IN T item)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PushFront(queue, &item));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PopFront, mPtr<mQueue<T>> &queue, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PopAt_Internal(queue, 0, pItem));

  ++queue->startIndex;
  --queue->count;

  if (queue->startIndex >= queue->size)
    queue->startIndex = 0;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PeekFront, mPtr<mQueue<T>> &queue, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PeekAt(queue, 0, pItem));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PeekBack, mPtr<mQueue<T>> &queue, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_PeekAt(queue, queue->count - 1, pItem));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PopBack, mPtr<mQueue<T>> &queue, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_PopAt_Internal(queue, queue->count - 1, pItem));

  --queue->count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PeekAt, mPtr<mQueue<T>> &queue, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count <= index, mR_IndexOutOfBounds);

  const size_t queueIndex = (queue->startIndex + index) % queue->size;

  *pItem = queue->pData[queueIndex];

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PopAt, mPtr<mQueue<T>> &queue, size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count <= index, mR_IndexOutOfBounds);

  if (index == 0)
  {
    mERROR_CHECK(mQueue_PopFront(queue, pItem));
    mRETURN_SUCCESS();
  }
  else if (index == queue->count - 1)
  {
    mERROR_CHECK(mQueue_PopBack(queue, pItem));
    mRETURN_SUCCESS();
  }

  const size_t queueIndex = (queue->startIndex + index) % queue->size;
  *pItem = std::move(queue->pData[queueIndex]);

  if (queue->startIndex + index < queue->size)
  {
    if (queueIndex + 1 != queue->size)
      mERROR_CHECK(mAllocator_Move(queue->pAllocator, &queue->pData[queueIndex], &queue->pData[queueIndex + 1], queue->size - (queue->startIndex + index)));

    if (queue->startIndex + queue->count > queue->size)
    {
      new (&queue->pData[queue->size - 1]) T(std::move(queue->pData[0]));

      if (queue->startIndex + queue->count - 1 > queue->size)
        mERROR_CHECK(mAllocator_Move(queue->pAllocator, &queue->pData[0], &queue->pData[1], queue->count - (queue->size - queue->startIndex)));
    }
  }
  else
  {
    mERROR_CHECK(mAllocator_Move(queue->pAllocator, &queue->pData[queueIndex], &queue->pData[queueIndex + 1], queue->count - (queue->size - queue->startIndex) - queueIndex - 1));
  }

  --queue->count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PointerAt, mPtr<mQueue<T>> &queue, const size_t index, OUT T **ppItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || ppItem == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count <= index, mR_IndexOutOfBounds);

  const size_t queueIndex = (queue->startIndex + index) % queue->size;

  *ppItem = &queue->pData[queueIndex];

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_GetCount, const mPtr<mQueue<T>> &queue, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = queue->count;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type* /* = nullptr */>
inline mFUNCTION(mQueue_Reserve, mPtr<mQueue<T>> &queue, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  if (count <= queue->size)
    mRETURN_SUCCESS();

  if (queue->pData == nullptr)
    mERROR_CHECK(mAllocator_Allocate(queue->pAllocator, &queue->pData, count));
  else
    mERROR_CHECK(mAllocator_Reallocate(queue->pAllocator, &queue->pData, count));

  queue->size = count;

  mRETURN_SUCCESS();
}

template<typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type* /* = nullptr */>
inline mFUNCTION(mQueue_Reserve, mPtr<mQueue<T>> &queue, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  if (count <= queue->size)
    mRETURN_SUCCESS();

  if (queue->pData == nullptr)
  {
    mERROR_CHECK(mAllocator_Allocate(queue->pAllocator, &queue->pData, count));
  }
  else
  {
    T *pNewData = nullptr;
    mDEFER(mAllocator_FreePtr(queue->pAllocator, &pNewData));
    mERROR_CHECK(mAllocator_Allocate(queue->pAllocator, &pNewData, count));

    if (queue->startIndex + queue->count <= queue->size)
    {
      mMoveConstructMultiple(&pNewData[queue->startIndex], &queue->pData[queue->startIndex], queue->count);
    }
    else
    {
      mMoveConstructMultiple(&pNewData[queue->startIndex], &queue->pData[queue->startIndex], queue->size - queue->startIndex);
      mMoveConstructMultiple(pNewData, queue->pData, queue->count - (queue->size - queue->startIndex));
    }

    std::swap(queue->pData, pNewData);
  }

  queue->size = count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Clear, mPtr<mQueue<T>> &queue)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  if (queue->pData != nullptr)
  {
    size_t index = queue->startIndex;

    for (size_t i = 0; i < queue->count; ++i)
    {
      mERROR_CHECK(mDestruct(queue->pData + index));

      index = (index + 1) % queue->size;
    }

    queue->startIndex = 0;
    queue->count = 0;
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_CopyTo, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(source == nullptr || pTarget == nullptr, mR_ArgumentNull);

  if (*pTarget == nullptr)
    mERROR_CHECK(mQueue_Create(pTarget, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pTarget));

  mERROR_CHECK(mQueue_Reserve(*pTarget, source->count));

  for (const auto &_item : source->Iterate())
    mERROR_CHECK(mQueue_PushBack(*pTarget, _item));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_OrderBy, mPtr<mQueue<T>> &queue, const std::function<mComparisonResult(const T &a, const T &b)> &orderByFunc)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || orderByFunc == nullptr, mR_ArgumentNull);

  struct
  {
    mQueue<T> *pQueue;

    void DualPivotQuickSort_Partition(int64_t low, int64_t high, int64_t *pRightPivot, int64_t *pLeftPivot, const std::function<mComparisonResult(const T &a, const T &b)> &orderByFunc)
    {
      if (mCR_Greater == orderByFunc((*pQueue)[low], (*pQueue)[high]))
        std::swap((*pQueue)[low], (*pQueue)[high]);

      int64_t j = low + 1;
      int64_t g = high - 1;
      int64_t k = low + 1;

      T *pQ = &(*pQueue)[low];
      T *pP = &(*pQueue)[high];

      while (k <= g)
      {
        if (mCR_Less == orderByFunc((*pQueue)[k], *pQ))
        {
          std::swap((*pQueue)[k], (*pQueue)[j]);
          j++;
        }

        else if (mCR_Less != orderByFunc((*pQueue)[k], *pP))
        {
          while (mCR_Greater == orderByFunc((*pQueue)[g], *pP) && k < g)
            g--;

          std::swap((*pQueue)[k], (*pQueue)[g]);
          g--;

          if (mCR_Less == orderByFunc((*pQueue)[k], *pQ))
          {
            std::swap((*pQueue)[k], (*pQueue)[j]);
            j++;
          }
        }

        k++;
      }

      j--;
      g++;

      std::swap((*pQueue)[low], (*pQueue)[j]);
      std::swap((*pQueue)[high], (*pQueue)[g]);

      *pLeftPivot = j;
      *pRightPivot = g;
    }

    void QuickSort(const int64_t start, const int64_t end, const std::function<mComparisonResult(const T &a, const T &b)> &orderByFunc)
    {
      if (start < end)
      {
        int64_t leftPivot, rightPivot;

        DualPivotQuickSort_Partition(start, end, &rightPivot, &leftPivot, orderByFunc);

        QuickSort(start, leftPivot - 1, orderByFunc);
        QuickSort(leftPivot + 1, rightPivot - 1, orderByFunc);
        QuickSort(rightPivot + 1, end, orderByFunc);
      }
    }

    void OrderBy(const std::function<mComparisonResult(const T &a, const T &b)> &orderByFunc)
    {
      QuickSort(0, (int64_t)pQueue->count - 1, orderByFunc);
    }

  } _internal;
  
  _internal.pQueue = queue.GetPointer();

  _internal.OrderBy(orderByFunc);

  mRETURN_SUCCESS();
}

template<typename T, typename TLessFunc, typename TGreaterFunc>
inline mFUNCTION(mQueue_OrderBy, mPtr<mQueue<T>> &queue)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  struct
  {
    mQueue<T> *pQueue;

    void DualPivotQuickSort_Partition(const int64_t low, const int64_t high, int64_t *pRightPivot, int64_t *pLeftPivot)
    {
      TLessFunc _less = TLessFunc();
      TGreaterFunc _greater = TGreaterFunc();

      if (_greater((*pQueue)[low], (*pQueue)[high]))
        std::swap((*pQueue)[low], (*pQueue)[high]);

      int64_t j = low + 1;
      int64_t g = high - 1;
      int64_t k = low + 1;

      T *pP = &(*pQueue)[low];
      T *pQ = &(*pQueue)[high];

      while (k <= g)
      {
        if (_less((*pQueue)[k], *pP))
        {
          std::swap((*pQueue)[k], (*pQueue)[j]);
          j++;
        }

        else if (!_less((*pQueue)[k], *pQ))
        {
          while (_greater((*pQueue)[g], *pQ) && k < g)
            g--;

          std::swap((*pQueue)[k], (*pQueue)[g]);
          g--;

          if (_less((*pQueue)[k], *pP))
          {
            std::swap((*pQueue)[k], (*pQueue)[j]);
            j++;
          }
        }

        k++;
      }

      j--;
      g++;

      std::swap((*pQueue)[low], (*pQueue)[j]);
      std::swap((*pQueue)[high], (*pQueue)[g]);

      *pLeftPivot = j;
      *pRightPivot = g;
    }

    void QuickSort(const int64_t start, const int64_t end)
    {
      if (start < end)
      {
        int64_t leftPivot, rightPivot;

        DualPivotQuickSort_Partition(start, end, &rightPivot, &leftPivot);

        QuickSort(start, leftPivot - 1);
        QuickSort(leftPivot + 1, rightPivot - 1);
        QuickSort(rightPivot + 1, end);
      }
    }

    void OrderBy()
    {
      QuickSort(0, (int64_t)pQueue->count - 1);
    }

  } _internal;

  _internal.pQueue = queue.GetPointer();

  _internal.OrderBy();

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Select, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator, const std::function<bool(const T &a)> &selectFunc)
{
  mFUNCTION_SETUP();

  mERROR_IF(source == nullptr || pTarget == nullptr || selectFunc == nullptr, mR_ArgumentNull);

  if (*pTarget == nullptr)
    mERROR_CHECK(mQueue_Create(pTarget, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pTarget));

  for (const auto &_item : source->Iterate())
  {
    if (selectFunc(_item))
      mERROR_CHECK(mQueue_PushBack(*pTarget, _item));
  }

  mRETURN_SUCCESS();
}

template<typename T, typename Func>
inline mFUNCTION(mQueue_Select, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator, Func func)
{
  mFUNCTION_SETUP();

  mERROR_IF(source == nullptr || pTarget == nullptr, mR_ArgumentNull);

  if (*pTarget == nullptr)
    mERROR_CHECK(mQueue_Create(pTarget, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pTarget));

  for (const auto &_item : source->Iterate())
  {
    if (func(_item))
      mERROR_CHECK(mQueue_PushBack(*pTarget, _item));
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Apply, mPtr<mQueue<T>> &queue, const std::function<mResult(T *pValue)> &func)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || func == nullptr, mR_ArgumentNull);

  for (auto &_item : queue->Iterate())
    mERROR_CHECK(func(&_item));

  mRETURN_SUCCESS();
}

template<typename T, typename Func>
inline mFUNCTION(mQueue_Apply, const mPtr<mQueue<T>> &queue)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || func == nullptr, mR_ArgumentNull);

  for (auto &_item : queue->Iterate())
    mERROR_CHECK(Func(&_item));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Min, const mPtr<mQueue<T>> &queue, const std::function<mComparisonResult(const T &a, const T &b)> &comparisonFunc, OUT T *pMin)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || comparisonFunc == nullptr || pMin == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count == 0, mR_ResourceStateInvalid);

  const T *pMinValue = &queue->pData[queue->startIndex];

  for (size_t i = 1; i < queue->count; i++)
  {
    T *pValue = &queue->pData[(queue->startIndex + i) % queue->size];

    if (comparisonFunc(*pValue, *pMinValue) == mCR_Less)
      pMinValue = pValue;
  }

  *pMin = *pMinValue;

  mRETURN_SUCCESS();
}

template<typename T, typename comparison>
inline mFUNCTION(mQueue_Min, const mPtr<mQueue<T>> &queue, OUT T *pMin, comparison _comparison)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pMin == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count == 0, mR_ResourceStateInvalid);

  const T *pMinValue = &queue->pData[queue->startIndex];

  for (size_t i = 1; i < queue->count; i++)
  {
    T *pValue = &queue->pData[(queue->startIndex + i) % queue->size];

    if (_comparison(*pValue, *pMinValue))
      pMinValue = pValue;
  }

  *pMin = *pMinValue;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Max, const mPtr<mQueue<T>> &queue, const std::function<mComparisonResult(const T &a, const T &b)> &comparisonFunc, OUT T *pMax)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || comparisonFunc == nullptr || pMax == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count == 0, mR_ResourceStateInvalid);

  const T *pMaxValue = &queue->pData[queue->startIndex];

  for (size_t i = 1; i < queue->count; i++)
  {
    T *pValue = &queue->pData[(queue->startIndex + i) % queue->size];

    if (comparisonFunc(*pValue, *pMaxValue) == mCR_Greater)
      pMaxValue = pValue;
  }

  *pMax = *pMaxValue;

  mRETURN_SUCCESS();
}

template<typename T, typename comparison>
inline mFUNCTION(mQueue_Max, const mPtr<mQueue<T>> &queue, OUT T *pMax, comparison _comparison)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pMax == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count == 0, mR_ResourceStateInvalid);

  const T *pMaxValue = &queue->pData[queue->startIndex];

  for (size_t i = 1; i < queue->count; i++)
  {
    T *pValue = &queue->pData[(queue->startIndex + i) % queue->size];

    if (_comparison(*pValue, *pMaxValue))
      pMaxValue = pValue;
  }

  *pMax = *pMaxValue;

  mRETURN_SUCCESS();
}

template<typename T, typename T2>
inline mFUNCTION(mQueue_Contains, const mPtr<mQueue<T>> &queue, const T2 &value, const std::function<bool(const T &a, const T2 &b)> &equalityComparer, OUT bool *pContained, OUT OPTIONAL size_t *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || equalityComparer == nullptr || pContained == nullptr, mR_ArgumentNull);

  size_t index = (size_t)-1;

  *pContained = false;

  for (const auto &_item : queue->Iterate())
  {
    ++index;

    if (equalityComparer(_item, value))
    {
      *pContained = true;

      if (pIndex != nullptr)
        *pIndex = index;

      break;
    }
  }

  mRETURN_SUCCESS();
}

template<typename T, typename T2, typename comparison>
inline mFUNCTION(mQueue_Contains, const mPtr<mQueue<T>> &queue, const T2 &value, OUT bool *pContained, OUT OPTIONAL size_t *pIndex, comparison _comparison)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pContained == nullptr, mR_ArgumentNull);

  size_t index = (size_t)-1;

  *pContained = false;

  for (const auto &_item : queue->Iterate())
  {
    ++index;

    if (_comparison(_item, value))
    {
      *pContained = true;

      if (pIndex != nullptr)
        *pIndex = index;

      break;
    }
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_RemoveDuplicates, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator, const std::function<bool(const T&a, const T&b)> &equalityComparer)
{
  mFUNCTION_SETUP();

  mERROR_IF(source == nullptr || pTarget == nullptr || equalityComparer == nullptr, mR_ArgumentNull);

  if (*pTarget == nullptr)
    mERROR_CHECK(mQueue_Create(pTarget, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pTarget));

  for (const auto &_item : source->Iterate())
  {
    bool contained;
    mERROR_CHECK(mQueue_Contains(*pTarget, _item, equalityComparer, &contained));

    if (!contained)
      mERROR_CHECK(mQueue_PushBack(*pTarget, _item));
  }

  mRETURN_SUCCESS();
}

template<typename T, typename comparison>
inline mFUNCTION(mQueue_RemoveDuplicates, const mPtr<mQueue<T>> &source, OUT mPtr<mQueue<T>> *pTarget, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(source == nullptr || pTarget == nullptr, mR_ArgumentNull);

  if (*pTarget == nullptr)
    mERROR_CHECK(mQueue_Create(pTarget, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pTarget));

  for (const auto &_item : source->Iterate())
  {
    bool contained;
    mERROR_CHECK((mQueue_Contains<T, T, comparison>(*pTarget, _item, &contained)));

    if (!contained)
      mERROR_CHECK(mQueue_PushBack(*pTarget, _item));
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Any, const mPtr<mQueue<T>> &queue, OUT bool *pAny)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pAny == nullptr, mR_ArgumentNull);

  *pAny = (queue->count != 0);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mQueue_Destroy_Internal, IN mQueue<T> *pQueue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pQueue == nullptr, mR_ArgumentNull);

  if (pQueue->pData != nullptr)
  {
    size_t index = pQueue->startIndex;

    for (size_t i = 0; i < pQueue->count; ++i)
    {
      mERROR_CHECK(mDestruct(pQueue->pData + index));

      index = (index + 1) % pQueue->size;
    }

    mAllocator *pAllocator = pQueue->pAllocator;
    mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pQueue->pData));
  }

  mRETURN_SUCCESS();
}

template<typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type* /* = nullptr */>
inline mFUNCTION(mQueue_Grow_Internal, mPtr<mQueue<T>> &queue)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  const size_t originalSize = queue->size;

  if (originalSize == 0)
  {
    mERROR_CHECK(mQueue_Reserve(queue, 1));
    mRETURN_SUCCESS();
  }

  const size_t doubleSize = originalSize * 2;

  mERROR_CHECK(mAllocator_Reallocate(queue->pAllocator, &queue->pData, doubleSize));

  queue->size = doubleSize;

  if (queue->startIndex > 0 && queue->startIndex + queue->count > originalSize)
  {
    const size_t wrappedCount = queue->count - (originalSize - queue->startIndex);
    mERROR_CHECK(mAllocator_Move(queue->pAllocator, &queue->pData[originalSize], &queue->pData[0], wrappedCount));
  }

  mRETURN_SUCCESS();
}

template<typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type* /* = nullptr */>
inline mFUNCTION(mQueue_Grow_Internal, mPtr<mQueue<T>> &queue)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  const size_t originalSize = queue->size;

  if (originalSize == 0)
  {
    mERROR_CHECK(mQueue_Reserve(queue, 1));
    mRETURN_SUCCESS();
  }

  const size_t doubleSize = originalSize * 2;

  T *pNewData = nullptr;
  mDEFER(mAllocator_FreePtr(queue->pAllocator, &pNewData));
  mERROR_CHECK(mAllocator_Allocate(queue->pAllocator, &pNewData, doubleSize));

  if (queue->startIndex + queue->count <= queue->size)
  {
    mMoveConstructMultiple(&pNewData[queue->startIndex], &queue->pData[queue->startIndex], queue->count);
  }
  else
  {
    mMoveConstructMultiple(&pNewData[queue->startIndex], &queue->pData[queue->startIndex], queue->size - queue->startIndex);
    mMoveConstructMultiple(pNewData, queue->pData, queue->count - (queue->size - queue->startIndex));
  }

  std::swap(queue->pData, pNewData);

  queue->size = doubleSize;

  if (queue->startIndex > 0 && queue->startIndex + queue->count > originalSize)
  {
    const size_t wrappedCount = queue->count - (originalSize - queue->startIndex);
    mERROR_CHECK(mAllocator_Move(queue->pAllocator, &queue->pData[originalSize], &queue->pData[0], wrappedCount));
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PopAt_Internal, mPtr<mQueue<T>> &queue, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(queue->count <= index, mR_IndexOutOfBounds);

  const size_t queueIndex = (queue->startIndex + index) % queue->size;

  *pItem = std::move(queue->pData[queueIndex]);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mQueueIterator<T>::mQueueIterator(const ptrdiff_t direction, T *pFirst, T *pLineEnd, T *pLineStart, const size_t count) :
  direction(direction),
  pCurrent(pFirst),
  pLineEnd(pLineEnd),
  pLineStart(pLineStart),
  pLastPos(pCurrent),
  count(count),
  index(0)
{ }

template<typename T>
inline T & mQueueIterator<T>::operator* ()
{
  return *pCurrent;
}

template<typename T>
inline const T & mQueueIterator<T>::operator*() const
{
  return *pCurrent;
}

template<typename T>
inline bool mQueueIterator<T>::operator!=(const mQueueIterator<T> &) const
{
  return index != count;
}

template<typename T>
inline mQueueIterator<T> & mQueueIterator<T>::operator++()
{
  pLastPos = pCurrent;
  pCurrent += direction;
  index++;

  if (direction > 0)
  {
    if (pCurrent > pLineEnd)
      pCurrent = pLineStart;
  }
  else
  {
    if (pCurrent < pLineStart)
      pCurrent = pLineEnd;
  }

  return *this;
}

template<typename T>
inline mConstQueueIterator<T>::mConstQueueIterator(const ptrdiff_t direction, const T *pFirst, const T *pLineEnd, const T *pLineStart, const size_t count) :
  mQueueIterator(direction, (T *)pFirst, (T *)pLineEnd, (T *)pLineStart, count)
{ }

template<typename T>
inline const T& mConstQueueIterator<T>::operator*() const
{
  return mQueueIterator<T>::operator *();
}

template<typename T>
inline bool mConstQueueIterator<T>::operator!=(const mConstQueueIterator<T> &iterator) const
{
  return mQueueIterator<T>::operator !=((const mQueueIterator<T> &)iterator);
}

template<typename T>
inline mConstQueueIterator<T>& mConstQueueIterator<T>::operator++()
{
  return (mConstQueueIterator<T> &)mQueueIterator<T>::operator ++();
}

#endif // mQueue_h__
