// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mRefPool.h"

template <typename T>
mFUNCTION(mRefPool_Destroy_Internal, mRefPool<T> *pRefPool);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mRefPool_Create, mPtr<mRefPool<T>> *pRefPool, mAllocator *pAllocator, const bool keepEntriesForever /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRefPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pRefPool, pAllocator, (std::function<void(mRefPool<T> *)>)[](mRefPool<T> *pData) { mRefPool_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mPool_Create(&(*pRefPool)->data, pAllocator));
  mERROR_CHECK(mPool_Create(&(*pRefPool)->ptrs, pAllocator));
  mERROR_CHECK(mMutex_Create(&(*pRefPool)->pMutex, pAllocator));
  (*pRefPool)->pAllocator = pAllocator;
  (*pRefPool)->keepForever = keepEntriesForever;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_Destroy, mPtr<mRefPool<T>> *pRefPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRefPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pRefPool));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_Add, mPtr<mRefPool<T>> &refPool, IN T *pItem, OUT mPtr<T> *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || pItem == nullptr || pIndex == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mRefPool_AddEmpty(refPool, pIndex));
  **pIndex = *pItem;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_AddEmpty, mPtr<mRefPool<T>> &refPool, OUT mPtr<T> *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || pIndex == nullptr, mR_ArgumentNull);

  size_t index;
  typename mRefPool<T>::refPoolPtrData empty;
  mERROR_CHECK(mPool_Add(refPool->data, &empty, &index));

  typename mRefPool<T>::refPoolPtrData *pPtrData = nullptr;
  mERROR_CHECK(mPool_PointerAt(refPool->data, index, &pPtrData));

  size_t ptrIndex;
  typename mRefPool<T>::refPoolPtr ptr;
  mERROR_CHECK(mPool_Add(refPool->ptrs, &ptr, &ptrIndex));

  pPtrData->index = ptrIndex;
  pPtrData->ptrParams.pUserData = &pPtrData->index;

  typename mRefPool<T>::refPoolPtr *pPtr = nullptr;
  mERROR_CHECK(mPool_PointerAt(refPool->ptrs, ptrIndex, &pPtr));

  pPtr->dataIndex = index;

  void *pRefPool = refPool.GetPointer();

#if defined (_DEBUG)
#define mRefPool_Internal_ERROR_CHECK(expr) do { mASSERT_DEBUG(mSUCCEEDED(expr), "Assertion Failed!"); if (mFAILED(expr)) return; } while (0)
#else
#define mRefPool_Internal_ERROR_CHECK(expr) do { if (mFAILED(expr)) return; } while (0)
#endif

  const std::function<void(T *)> &destructionFunction = [pRefPool, index](T *pData)
  {
    mPtr<mRefPool<T>> _refPool;
    mDEFER_DESTRUCTION(&_refPool, mSharedPointer_Destroy);

    mRefPool_Internal_ERROR_CHECK(mSharedPointer_Create(&_refPool, (mRefPool<T> *)pRefPool, mSHARED_POINTER_FOREIGN_RESOURCE));
    mRefPool_Internal_ERROR_CHECK(mMutex_Lock(_refPool->pMutex));

    mDEFER_DESTRUCTION(_refPool->pMutex, mMutex_Unlock);

    typename mRefPool<T>::refPoolPtrData *_pPtrData = nullptr;
    mRefPool_Internal_ERROR_CHECK(mPool_PointerAt(_refPool->data, index, &_pPtrData));

    _pPtrData->ptrParams.cleanupFunction = nullptr;

    mASSERT(pData == &_pPtrData->element, "Reference pool corruption detected.");
    mRefPool_Internal_ERROR_CHECK(mDestruct(pData));

    typename mRefPool<T>::refPoolPtrData data;
    mRefPool_Internal_ERROR_CHECK(mPool_RemoveAt(_refPool->data, index, &data));

    typename mRefPool<T>::refPoolPtr *_pPtr;
    mRefPool_Internal_ERROR_CHECK(mPool_PointerAt(_refPool->ptrs, data.index, &_pPtr));

    _pPtr->ptr.m_pData = nullptr;

    typename mRefPool<T>::refPoolPtr _ptr;
    mRefPool_Internal_ERROR_CHECK(mPool_RemoveAt(_refPool->ptrs, data.index, &_ptr));
  };

  mERROR_CHECK(mSharedPointer_CreateInplace(&pPtr->ptr, &pPtrData->ptrParams, &pPtrData->element, mSHARED_POINTER_FOREIGN_RESOURCE, destructionFunction));

  *pIndex = pPtr->ptr;

  if (!refPool->keepForever)
    --pPtrData->ptrParams.referenceCount;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_ForEach, mPtr<mRefPool<T>> &refPool, const std::function<mResult(mPtr<T> &)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || function == nullptr, mR_ArgumentNull);

  const std::function<mResult (typename mRefPool<T>::refPoolPtr *, size_t)> fn = [function](typename mRefPool<T>::refPoolPtr *pData, size_t) 
  { 
    RETURN function(pData->ptr); 
  };

  mERROR_CHECK(mPool_ForEach(refPool->ptrs, fn));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mDestruct, struct mRefPool<T>::refPoolPtr *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_Success);

  mERROR_CHECK(mSharedPointer_Destroy(&pData->ptr));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mRefPool_Destroy_Internal, mRefPool<T> *pRefPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRefPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mPool_Destroy(&pRefPool->ptrs));
  mERROR_CHECK(mPool_Destroy(&pRefPool->data));
  mERROR_CHECK(mMutex_Destroy(&pRefPool->pMutex));

  mRETURN_SUCCESS();
}
