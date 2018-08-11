// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mQueue_h__
#define mQueue_h__

#include "default.h"

template <typename T>
struct mQueue
{
  mAllocator *pAllocator;
  T *pData;
  size_t startIndex, size, count;
};

template <typename T>
mFUNCTION(mQueue_Create, OUT mPtr<mQueue<T>> *pQueue, IN OPTIONAL mAllocator *pAllocator);

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
mFUNCTION(mQueue_GetCount, mPtr<mQueue<T>> &queue, OUT size_t *pCount);

template <typename T>
mFUNCTION(mQueue_Reserve, mPtr<mQueue<T>> &queue, const size_t count);

//////////////////////////////////////////////////////////////////////////

template<typename T>
mFUNCTION(mQueue_Destroy_Internal, IN mQueue<T> *pQueue);

template<typename T>
mFUNCTION(mQueue_Grow_Internal, mPtr<mQueue<T>> &queue);

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

  mDEFER_DESTRUCTION_ON_ERROR(pQueue, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create(pQueue, pQueueRaw, (std::function<void (mQueue<T> *)>)[](mQueue<T> *pData) { mQueue_Destroy_Internal<T>(pData); }, pAllocator));
  pQueueRaw = nullptr;

  (*pQueue)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_Destroy, IN_OUT mPtr<mQueue<T>> *pQueue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pQueue == nullptr || *pQueue == nullptr, mR_ArgumentNull);
  
  mERROR_CHECK(mSharedPointer_Destroy(pQueue));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PushBack, mPtr<mQueue<T>> &queue, IN T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  if (queue->pData == nullptr)
    mERROR_CHECK(mQueue_Reserve(queue, 1));
  else if (queue->size == queue->count)
    mERROR_CHECK(mQueue_Grow_Internal(queue));

  const size_t index = (queue->startIndex + queue->count) % queue->size;
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
inline mFUNCTION(mQueue_PushFront, mPtr<mQueue<T>>& queue, IN T * pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr, mR_ArgumentNull);

  if (queue->pData == nullptr)
    mERROR_CHECK(mQueue_Reserve(queue, 1));
  else if (queue->size == queue->count)
    mERROR_CHECK(mQueue_Grow_Internal(queue));

  size_t index;

  if (queue->startIndex > 0)
    index = queue.startIndex - 1;
  else
    index = queue.size - 1;

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

  mERROR_CHECK(mQueue_PeekFront(queue, pItem));

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
inline mFUNCTION(mQueue_PeekBack, mPtr<mQueue<T>>& queue, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PeekAt(queue, queue->count - 1, pItem));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PopBack, mPtr<mQueue<T>> &queue, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_PeekBack(queue, pItem));

  --queue->count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_PeekAt, mPtr<mQueue<T>> &queue, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);

  if (queue->count < index)
    mRETURN_RESULT(mR_IndexOutOfBounds);

  const size_t queueIndex = (queue->startIndex + index) % queue->size;

  *pItem = queue->pData[queueIndex];

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mQueue_GetCount, mPtr<mQueue<T>> &queue, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(queue == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = queue->count;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
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

template<typename T>
inline mFUNCTION(mQueue_Destroy_Internal, IN mQueue<T> *pQueue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pQueue == nullptr, mR_ArgumentNull);

  if (pQueue->pData != nullptr)
  {
    mAllocator *pAllocator = pQueue->pAllocator;
    mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pQueue->pData));
  }

  mRETURN_SUCCESS();
}

template<typename T>
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

#endif // mQueue_h__
