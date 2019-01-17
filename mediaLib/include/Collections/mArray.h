#ifndef mArray_h__
#define mArray_h__

#include "mediaLib.h"

template <typename T>
struct mArray
{
  mArray() { }

  T *pData = nullptr;
  size_t count = 0;
  mAllocator *pAllocator = nullptr;

  T& operator [](const size_t index);
};

template <typename T>
mFUNCTION(mArray_Create, OUT mArray<T> *pArray, mAllocator *pAllocator, const size_t count);

template <typename T>
mFUNCTION(mArray_Destroy, IN_OUT mArray<T> *pArray);

template <typename T>
mFUNCTION(mArray_Resize, mArray<T> &arrayRef, const size_t newCount);

template <typename T>
mFUNCTION(mArray_PopAt, mArray<T> &arrayRef, const size_t index, OUT T *pData);

template <typename T>
mFUNCTION(mArray_PeekAt, mArray<T> &arrayRef, const size_t index, OUT T *pData);

template <typename T>
mFUNCTION(mArray_PutAt, mArray<T> &arrayRef, const size_t index, IN T *pData);

template <typename T>
mFUNCTION(mArray_PutDataAt, mArray<T> &arrayRef, const size_t index, T pData);

template <typename T>
mFUNCTION(mArray_GetPointer, mArray<T> &arrayRef, OUT T **ppData);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mArray_Create, OUT mArray<T> *pArray, mAllocator *pAllocator, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pArray == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pArray->pData, count));

  pArray->count = count;
  pArray->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mArray_Destroy, IN_OUT mArray<T> *pArray)
{
  mFUNCTION_SETUP();

  mERROR_IF(pArray == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_FreePtr(pArray->pAllocator, &pArray->pData));

  pArray->count = 0;
  pArray->pAllocator = nullptr;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mArray_Resize, mArray<T> &arrayRef, const size_t newCount)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAllocator_Reallocate(arrayRef->pAllocator, &arrayRef->pData, newCount));

  if (newCount > count)
    mERROR_CHECK(mMemset(&arrayRef.pData[arrayRef.count], newCount - count));

  arrayRef.count = newCount;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mArray_PopAt, mArray<T> &arrayRef, const size_t index, OUT T *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(arrayRef.pData == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(arrayRef.count <= index, mR_IndexOutOfBounds);

  *pData = std::move(arrayRef[index]);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mArray_PeekAt, mArray<T> &arrayRef, const size_t index, OUT T *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(arrayRef.pData == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(arrayRef.count <= index, mR_IndexOutOfBounds);

  *pData = arrayRef[index];

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mArray_PutAt, mArray<T> &arrayRef, const size_t index, IN T *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(arrayRef.pData == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(arrayRef.count <= index, mR_IndexOutOfBounds);

  arrayRef.pData[index] = *pData;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mArray_PutDataAt, mArray<T> &arrayRef, const size_t index, T pData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mArray_PutAt(arrayRef, index, &pData));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mArray_GetPointer, mArray<T> &arrayRef, OUT T **ppData)
{
  mFUNCTION_SETUP();

  mERROR_IF(arrayRef.pData == nullptr || ppData == nullptr, mR_ArgumentNull);

  *ppData = arrayRef.pData;

  mRETURN_SUCCESS();
}

template<typename T>
inline T& mArray<T>::operator[](const size_t index)
{
  return pData[index];
}

#endif // mArray_h__
