#ifndef mList_h__
#define mList_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "xSX9SVBtDY7Bydi/1+D7FA+adjXc6qisEwIzG2ZlrttVOfpYgkbB4jsUoGjz2hS8TQLhHiHMx9jZLV82"
#endif

//////////////////////////////////////////////////////////////////////////

template <typename T>
struct mListIterator
{
  ptrdiff_t direction;
  T *pCurrent;

  mListIterator(const ptrdiff_t direction, T *pPtr);
  T &operator *();
  const T &operator *() const;
  bool operator != (T *pEnd) const;
  void operator++();
};

template <typename T>
struct mConstListIterator
{
  ptrdiff_t direction;
  const T *pCurrent;

  mConstListIterator(const ptrdiff_t direction, const T *pPtr);
  const T &operator *() const;
  bool operator != (const T *pEnd) const;
  void operator++();
};

//////////////////////////////////////////////////////////////////////////

template <typename T>
struct mList
{
  mAllocator *pAllocator = nullptr;
  T *pData = nullptr;
  size_t count = 0, capacity = 0;

  mListIterator<T> begin()
  {
    return mListIterator<T>(1, pData);
  }

  T * end()
  {
    return pData + count;
  };

  mConstListIterator<T> begin() const
  {
    return mConstListIterator<T>(1, pData);
  }

  const T * end() const
  {
    return pData + count;
  };

  struct mListReverseIterator
  {
    mList<T> *pList;

    mListIterator<T> begin() { return mListIterator(-1, pList->pData + pList->count - 1); }
    T * end() { return pList->pData - 1; }

  } IterateReverse() { return { this }; };

  struct mListConstReverseIterator
  {
    const mList<T> *pList;

    mConstListIterator<T> begin() { return mConstListIterator(-1, pList->pData + pList->count - 1); }
    T *end() { return pList->pData - 1; }

  } IterateReverse() const { return { this }; };

  struct mListIteratorWrapper
  {
    mList<T> *pList;

    mListIterator<T> begin() { return mListIterator<T>(1, pList->pData); }
    T * end() { return pList->pData + pList->count; }

  } Iterate() { return { this }; };

  struct mListConstIteratorWrapper
  {
    const mList<T> *pList;

    mConstListIterator<T> begin() const { return mConstListIterator<T>(1, pList->pData); }
    const T *end() { return pList->pData + pList->count; }

  } Iterate() const { return { this }; };

  inline T &operator[](const size_t index)
  {
    return pData[index];
  }

  inline const T &operator[](const size_t index) const
  {
    return pData[index];
  }

public:
  ~mList();
};

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mList_Create, OUT mList<T> *pList, IN OPTIONAL mAllocator *pAllocator);

template <typename T>
mFUNCTION(mList_Create, OUT mPtr<mList<T>> *pList, IN OPTIONAL mAllocator *pAllocator);

template <typename T>
mFUNCTION(mList_Create, OUT mUniqueContainer<mList<T>> *pList, IN OPTIONAL mAllocator *pAllocator);

template <typename T>
mFUNCTION(mList_Destroy, IN_OUT mPtr<mList<T>> *pList);

template <typename T>
mFUNCTION(mList_Destroy, IN_OUT mList<T> *pList);

template <typename T>
mFUNCTION(mList_Clear, mList<T> &list);

template<typename T>
mFUNCTION(mList_GetCount, const mList<T> &list, OUT size_t *pCount);

template <typename T, typename std::enable_if<mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mList_Reserve, mList<T> &list, const size_t count);

template <typename T, typename std::enable_if<!mIsTriviallyMemoryMovable<T>::value>::type* = nullptr>
mFUNCTION(mList_Reserve, mList<T> &list, const size_t count);

template<typename T>
inline mFUNCTION(mList_ResizeWith, mList<T> &list, const size_t count, const T &value);

template<typename T>
inline mFUNCTION(mList_PushBack, mList<T> &list, IN const T *pItem);

template<typename T>
inline mFUNCTION(mList_PushBack, mList<T> &list, IN const T &item);

template<typename T>
inline mFUNCTION(mList_PushBack, mList<T> &list, IN T &&item);

template <typename T, typename ...Args>
mFUNCTION(mList_EmplaceBack, mList<T> &list, Args && ...args);

template<typename T>
inline mFUNCTION(mList_PopAt, mList<T> &list, size_t index, OUT T *pItem);

template<typename T>
inline mFUNCTION(mList_InsertAt, mList<T> &list, size_t index, const T *pItem);

template<typename T>
inline mFUNCTION(mList_InsertAt, mList<T> &list, size_t index, const T &item);

template<typename T>
inline mFUNCTION(mList_InsertAt, mList<T> &list, size_t index, T &&item);

template<typename T>
inline mFUNCTION(mList_PopBack, mList<T> &list, OUT T *pItem);

template<typename T>
inline mFUNCTION(mList_PeekAt, mList<T> &list, const size_t index, OUT T *pItem);

template<typename T>
inline mFUNCTION(mList_PointerAt, mList<T> &list, const size_t index, OUT T **ppItem);

template<typename T>
inline mFUNCTION(mList_PointerAt, const mList<T> &list, const size_t index, OUT T * const *ppItem);

//////////////////////////////////////////////////////////////////////////

#include "mList.inl"

#endif mList_h__
