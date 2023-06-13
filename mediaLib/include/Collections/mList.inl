#include "mList.h"

//////////////////////////////////////////////////////////////////////////

template <typename T>
inline void mList_Destroy_Internal(mList<T> *pList)
{
  if (pList == nullptr)
    return;

  mList_Clear(*pList);

  mAllocator_FreePtr(pList->pAllocator, &pList->pData);
  pList->capacity = 0;
}

template<typename T>
inline mList<T>::~mList()
{
  mList_Destroy_Internal(this);
}

//////////////////////////////////////////////////////////////////////////

template <typename T>
inline mFUNCTION(mList_Create, OUT mList<T> *pList, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pList == nullptr, mR_ArgumentNull);

  mList_Destroy_Internal(pList);

  pList->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mList_Create, OUT mPtr<mList<T>> *pList, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pList == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pList, pAllocator, mList_Destroy_Internal<T>, 1));

  (*pList)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mList_Create, OUT mUniqueContainer<mList<T>> *pList, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pList == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mUniqueContainer<mList<T>>::CreateWithCleanupFunction(pList, pAllocator, mList_Destroy_Internal<T>, 1));

  (*pList)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mList_Destroy, IN_OUT mPtr<mList<T>> *pList)
{
  return mSharedPointer_Destroy(pList);
}

template <typename T>
inline mFUNCTION(mList_Destroy, IN_OUT mList<T> *pList)
{
  mList_Destroy_Internal(pList);

  return mR_Success;
}

template <typename T>
mFUNCTION(mList_Clear, mList<T> &list)
{
  mFUNCTION_SETUP();

  if (list.pData != nullptr)
  {
    for (size_t i = 0; i < list.count; ++i)
      mERROR_CHECK(mDestruct(list.pData + i));

    list.count = 0;
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_GetCount, const mList<T> &list, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCount == nullptr, mR_ArgumentNull);

  *pCount = list.count;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mList_Grow_Internal, mList<T> &queue);

template<typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mList_Grow_Internal, mList<T> &queue);

//////////////////////////////////////////////////////////////////////////

template<typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type * /* = nullptr */>
inline mFUNCTION(mList_Grow_Internal, mList<T> &list)
{
  mFUNCTION_SETUP();

  const size_t originalSize = list.capacity;

  if (originalSize == 0)
  {
    mERROR_CHECK(mList_Reserve(list, 1));
    mRETURN_SUCCESS();
  }

  const size_t doubleSize = originalSize * 2;

  mERROR_CHECK(mAllocator_Reallocate(list.pAllocator, &list.pData, doubleSize));

  list.capacity = doubleSize;

  mRETURN_SUCCESS();
}

template<typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type * /* = nullptr */>
inline mFUNCTION(mList_Grow_Internal, mList<T> &list)
{
  mFUNCTION_SETUP();

  const size_t originalSize = list.capacity;

  if (originalSize == 0)
  {
    mERROR_CHECK(mList_Reserve(list, 1));
    mRETURN_SUCCESS();
  }

  const size_t doubleSize = originalSize * 2;

  T *pNewData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, list.pAllocator, &pNewData);
  mERROR_CHECK(mAllocator_Allocate(list.pAllocator, &pNewData, doubleSize));

  mMoveConstructMultiple(pNewData, list.pData, list.count);

  std::swap(list.pData, pNewData);

  list.capacity = doubleSize;

  mRETURN_SUCCESS();
}

template<typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type * /* = nullptr */>
inline mFUNCTION(mList_Reserve, mList<T> &list, const size_t count)
{
  mFUNCTION_SETUP();

  if (count <= list.capacity)
    mRETURN_SUCCESS();

  if (list.pData == nullptr)
    mERROR_CHECK(mAllocator_Allocate(list.pAllocator, &list.pData, count));
  else
    mERROR_CHECK(mAllocator_Reallocate(list.pAllocator, &list.pData, count));

  list.capacity = count;

  mRETURN_SUCCESS();
}

template<typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type * /* = nullptr */>
inline mFUNCTION(mList_Reserve, mList<T> &list, const size_t count)
{
  mFUNCTION_SETUP();

  if (count <= list.size)
    mRETURN_SUCCESS();

  if (list.pData == nullptr)
  {
    mERROR_CHECK(mAllocator_Allocate(list.pAllocator, &list.pData, count));
  }
  else
  {
    T *pNewData = nullptr;
    mDEFER_CALL_2(mAllocator_FreePtr, list.pAllocator, &pNewData);
    mERROR_CHECK(mAllocator_Allocate(list.pAllocator, &pNewData, count));

    mMoveConstructMultiple(pNewData, list.pData, list.count);

    std::swap(list.pData, pNewData);
  }

  list.capacity = count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_ResizeWith, mList<T> &list, const size_t count, const T &value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mList_Reserve(list, count));

  if (list.count < count)
  {
    for (size_t i = list.count; i < count; i++)
      new (&list.pData[i]) T(value);

    list.count = count;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mList_PushBack, mList<T> &list, IN const T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pItem == nullptr, mR_ArgumentNull);

  if (list.pData == nullptr)
    mERROR_CHECK(mList_Reserve(list, 1));
  else if (list.capacity == list.count)
    mERROR_CHECK(mList_Grow_Internal(list));

  if (!std::is_trivially_copy_constructible<T>::value)
    new (&list.pData[list.count]) T(*pItem);
  else
    mERROR_CHECK(mMemcpy(&list.pData[list.count], pItem, 1));

  ++list.count;

  mRETURN_SUCCESS();
}

template <typename T, typename ...Args>
mFUNCTION(mList_EmplaceBack, mList<T> &list, Args && ...args)
{
  mFUNCTION_SETUP();

  if (list.pData == nullptr)
    mERROR_CHECK(mList_Reserve(list, 1));
  else if (list.capacity == list.count)
    mERROR_CHECK(mList_Grow_Internal(list));

  new (&list.pData[list.count]) T(std::forward<Args>(args)...);

  ++list.count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_PushBack, mList<T> &list, IN const T &item)
{
  mFUNCTION_SETUP();

  mERROR_CHECK((mList_PushBack<T>(list, &item)));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_PushBack, mList<T> &list, IN T &&item)
{
  return mList_EmplaceBack(list, std::forward<T>(item));
}

template<typename T>
inline mFUNCTION(mList_PopAt, mList<T> &list, size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(list.count <= index, mR_IndexOutOfBounds);

  *pItem = std::move(list.pData[index]);

  if (index + 1 != list.count)
    mERROR_CHECK(mMove(&list.pData[index], &list.pData[index + 1], list.count - index - 1));

  --list.count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_InsertAt, mList<T> &list, size_t index, const T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(list.count < index, mR_IndexOutOfBounds);

  if (list.pData == nullptr)
    mERROR_CHECK(mList_Reserve(list, 1));
  else if (list.capacity == list.count)
    mERROR_CHECK(mList_Grow_Internal(list));

  if (index == list.count)
  {
    mERROR_CHECK(mList_PushBack(list, pItem));
    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mMove(&list.pData[index + 1], &list.pData[index], list.count - index));

  new (&list.pData[index]) T(*pItem);

  ++list.count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_InsertAt, mList<T> &list, size_t index, const T &item)
{
  return mList_InsertAt(list, index, &item);
}

template<typename T>
inline mFUNCTION(mList_InsertAt, mList<T> &list, size_t index, T &&item)
{
  mFUNCTION_SETUP();

  mERROR_IF(list.count < index, mR_IndexOutOfBounds);

  if (list.pData == nullptr)
    mERROR_CHECK(mList_Reserve(list, 1));
  else if (list.capacity == list.count)
    mERROR_CHECK(mList_Grow_Internal(list));
  
  if (index == list.count)
  {
    mERROR_CHECK(mList_PushBack(list, item));
    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mMove(&list.pData[index + 1], &list.pData[index], list.count - index));

  new (&list.pData[index]) T(item);

  ++list.count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_PopBack, mList<T> &list, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(list.count == 0, mR_ResourceStateInvalid);

  *pItem = std::move(list.pData[list.count - 1]);

  --list.count;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_PeekAt, const mList<T> &list, const size_t index, OUT T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(list.count <= index, mR_IndexOutOfBounds);

  *pItem = list.pData[index];

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_PointerAt, mList<T> &list, const size_t index, OUT T **ppItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppItem == nullptr, mR_ArgumentNull);
  mERROR_IF(list.count <= index, mR_IndexOutOfBounds);

  *ppItem = &list.pData[index];

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mList_PointerAt, const mList<T> &list, const size_t index, OUT T * const *ppItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppItem == nullptr, mR_ArgumentNull);
  mERROR_IF(list.count <= index, mR_IndexOutOfBounds);

  *ppItem = &list.pData[index];

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mListIterator<T>::mListIterator(const ptrdiff_t direction, T *pPtr) :
  direction(direction),
  pCurrent(pPtr) { }

template <typename T>
inline T & mListIterator<T>::operator *()
{
  return *pCurrent;
}

template <typename T>
inline const T & mListIterator<T>::operator *() const
{
  return *pCurrent;
}

template <typename T>
inline bool mListIterator<T>::operator != (T *pEnd) const
{
  return pCurrent != pEnd;
}

template<typename T>
inline void mListIterator<T>::operator++()
{
  pCurrent += direction;
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mConstListIterator<T>::mConstListIterator(const ptrdiff_t direction, const T *pPtr) :
  direction(direction),
  pCurrent(pPtr) { }

template <typename T>
inline const T &mConstListIterator<T>::operator *() const
{
  return *pCurrent;
}

template <typename T>
inline bool mConstListIterator<T>::operator != (const T *pEnd) const
{
  return pCurrent != pEnd;
}

template<typename T>
inline void mConstListIterator<T>::operator++()
{
  pCurrent += direction;
}
