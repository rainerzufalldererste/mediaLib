#include "mRefPool.h"
#include "mQueue.h"

template <typename T>
mFUNCTION(mRefPool_Destroy_Internal, IN mRefPool<T> *pRefPool);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mRefPool_Create, OUT mPtr<mRefPool<T>> *pRefPool, IN mAllocator *pAllocator, const bool keepEntriesForever /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRefPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pRefPool, pAllocator, (std::function<void(mRefPool<T> *)>)[](mRefPool<T> *pData) { mRefPool_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mPool_Create(&(*pRefPool)->data, pAllocator));
  mERROR_CHECK(mPool_Create(&(*pRefPool)->ptrs, pAllocator));
  mERROR_CHECK(mRecursiveMutex_Create(&(*pRefPool)->pMutex, pAllocator));
  (*pRefPool)->pAllocator = pAllocator;
  (*pRefPool)->keepForever = keepEntriesForever;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_Destroy, IN_OUT mPtr<mRefPool<T>> *pRefPool)
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

  if (std::is_trivially_copy_constructible<T>::value)
    new (pIndex->GetPointer()) T(*pItem);
  else
    mERROR_CHECK(mAllocator_Copy(refPool->pAllocator, pIndex->GetPointer(), pItem, 1));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_AddEmpty, mPtr<mRefPool<T>> &refPool, OUT mPtr<T> *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || pIndex == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mRecursiveMutex_Lock(refPool->pMutex));
  mDEFER_CALL(refPool->pMutex, mRecursiveMutex_Unlock);

  size_t index;
  typename mRefPool<T>::refPoolPtrData empty;
  mERROR_CHECK(mPool_Add(refPool->data, &empty, &index));

  typename mRefPool<T>::refPoolPtrData *pPtrData = nullptr;
  mERROR_CHECK(mPool_PointerAt(refPool->data, index, &pPtrData));

  size_t ptrIndex;
  mRefPool_SharedPointerContainer_Internal<T> ptr;
  mERROR_CHECK(mPool_Add(refPool->ptrs, &ptr, &ptrIndex));

  pPtrData->index = ptrIndex;

  mRefPool_SharedPointerContainer_Internal<T> *pPtr = nullptr;
  mERROR_CHECK(mPool_PointerAt(refPool->ptrs, ptrIndex, &pPtr));

  pPtr->dataIndex = index;

  void *pRefPool = refPool.GetPointer();

#if defined (_DEBUG)
#define mRefPool_Internal_ERROR_CHECK(expr) do { mResult __result = (expr); mASSERT_DEBUG(mSUCCEEDED(__result), "Assertion Failed! [Result is %" PRIi32 "]", __result); if (mFAILED(__result)) return; } while (0)
#else
#define mRefPool_Internal_ERROR_CHECK(expr) do { if (mFAILED(expr)) return; } while (0)
#endif

  const std::function<void(T *)> &destructionFunction = [pRefPool, index](T *pData)
  {
    mPtr<mRefPool<T>> _refPool;
    mDEFER_CALL(&_refPool, mSharedPointer_Destroy);

    mRefPool_Internal_ERROR_CHECK(mSharedPointer_Create(&_refPool, (mRefPool<T> *)pRefPool, mSHARED_POINTER_FOREIGN_RESOURCE));

    mRefPool_Internal_ERROR_CHECK(mRecursiveMutex_Lock(_refPool->pMutex));
    mDEFER_CALL(_refPool->pMutex, mRecursiveMutex_Unlock);

    typename mRefPool<T>::refPoolPtrData *_pPtrData = nullptr;
    mRefPool_Internal_ERROR_CHECK(mPool_PointerAt(_refPool->data, index, &_pPtrData));

    mASSERT(pData == &_pPtrData->element, "Reference pool corruption detected.");
    mRefPool_Internal_ERROR_CHECK(mDestruct(pData));

    typename mRefPool<T>::refPoolPtrData data;
    mRefPool_Internal_ERROR_CHECK(mPool_RemoveAt(_refPool->data, index, &data));

    mRefPool_SharedPointerContainer_Internal<T> *_pPtr;
    mRefPool_Internal_ERROR_CHECK(mPool_PointerAt(_refPool->ptrs, data.index, &_pPtr));

    _pPtr->ptr.m_pData = nullptr;

    mRefPool_SharedPointerContainer_Internal<T> _ptr;
    mRefPool_Internal_ERROR_CHECK(mPool_RemoveAt(_refPool->ptrs, data.index, &_ptr));
  };

  mERROR_CHECK(mSharedPointer_CreateInplace(&pPtr->ptr, &pPtrData->ptrParams, &pPtrData->element, mSHARED_POINTER_FOREIGN_RESOURCE, destructionFunction));

  pPtrData->ptrParams.pUserData = &pPtrData->index;

  *pIndex = pPtr->ptr;

  if (!refPool->keepForever)
    --pPtrData->ptrParams.referenceCount;

  mERROR_CHECK(mMemset(pIndex->GetPointer(), 1));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_Crush, mPtr<mRefPool<T>> &refPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mRecursiveMutex_Lock(refPool->pMutex));
  mDEFER_CALL(refPool->pMutex, mRecursiveMutex_Unlock);

  mPtr<mPool<mRefPool_SharedPointerContainer_Internal<T>>> crushedPtrs;
  mDEFER_CALL(&crushedPtrs, mPool_Destroy);
  mERROR_CHECK(mPool_Create(&crushedPtrs, refPool->pAllocator));

  for (auto _item : refPool->ptrs->Iterate())
  {
    mERROR_IF(_item.pData->ptr.m_pParams->pUserData == nullptr, mR_InternalError);

    size_t newIndex;
    mERROR_CHECK(mPool_Add(crushedPtrs, _item.pData, &newIndex));
    *(size_t *)_item.pData->ptr.m_pParams->pUserData = newIndex;
  };
  
  mERROR_CHECK(mPool_Destroy(&refPool->ptrs));
  refPool->ptrs = crushedPtrs;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_ForEach, mPtr<mRefPool<T>> &refPool, const std::function<mResult(mPtr<T> &)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || function == nullptr, mR_ArgumentNull);

  for (auto _item : refPool->ptrs->Iterate())
  {
    mERROR_IF(_item.pData->ptr == nullptr, mR_Success);

    const mResult result = (function(_item.pData->ptr));

    if (mFAILED(result))
    {
      if (result == mR_Break)
        break;

      mRETURN_RESULT(result);
    }
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_KeepIfTrue, mPtr<mRefPool<T>> &refPool, const std::function<mResult(mPtr<T>&, OUT bool*)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || function == nullptr, mR_ArgumentNull);
  mERROR_IF(!refPool->keepForever, mR_ResourceStateInvalid);

  mPtr<mQueue<size_t>> removeIndexes;
  mDEFER_CALL(&removeIndexes, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&removeIndexes, &mDefaultTempAllocator));

  for (auto _item : refPool->ptrs->Iterate())
  {
    bool keep = true;

    const mResult result = function(_item.pData->ptr, &keep);

    if (mFAILED(result))
    {
      if (result == mR_Break)
        break;
      else
        mRETURN_RESULT(result);
    }

    if (!keep)
      mERROR_CHECK(mQueue_PushBack(removeIndexes, _item.index));
  };

  size_t removeCount = 0;
  mERROR_CHECK(mQueue_GetCount(removeIndexes, &removeCount));
  
  for (size_t i = 0; i < removeCount; i++)
  {
    size_t index;
    mERROR_CHECK(mQueue_PopFront(removeIndexes, &index));

    mRefPool_SharedPointerContainer_Internal<T> *pData;
    mERROR_CHECK(mPool_PointerAt(refPool->ptrs, index, &pData));

    if (pData->ptr.m_pParams->referenceCount == 1)
      mERROR_CHECK(mSharedPointer_Destroy(&pData->ptr));
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_PeekAt, mPtr<mRefPool<T>> &refPool, const size_t index, OUT mPtr<T> *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || pIndex == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mRecursiveMutex_Lock(refPool->pMutex));
  mDEFER_CALL(refPool->pMutex, mRecursiveMutex_Unlock);

  mRefPool_SharedPointerContainer_Internal<T> refPoolPointer;
  mERROR_CHECK(mPool_PeekAt(refPool->ptrs, index, &refPoolPointer));

  *pIndex = refPoolPointer.ptr;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_GetCount, mPtr<mRefPool<T>> &refPool, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr || pCount == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mPool_GetCount(refPool->ptrs, pCount));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_RemoveOwnReference, mPtr<mRefPool<T>> &refPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(refPool == nullptr, mR_ArgumentNull);
  mERROR_IF(!refPool->keepForever, mR_ResourceStateInvalid);
  refPool->keepForever = false;

  mERROR_CHECK(mRecursiveMutex_Lock(refPool->pMutex));
  mDEFER_CALL(refPool->pMutex, mRecursiveMutex_Unlock);

  for (auto _item : refPool->ptrs->Iterate())
  {
    mERROR_IF(_item.pData == nullptr || _item.pData->ptr == nullptr, mR_ArgumentNull);

    if (_item.pData->ptr.m_pParams->referenceCount > 1)
      --_item.pData->ptr.m_pParams->referenceCount;
    else
      mERROR_CHECK(mSharedPointer_Destroy(&_item.pData->ptr));
  };

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mRefPool_GetPointerIndex, const mPtr<T> &ptr, size_t *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pIndex == nullptr || ptr == nullptr, mR_ArgumentNull);
  mERROR_IF(ptr.m_pParams->pUserData == nullptr, mR_ResourceStateInvalid);

  *pIndex = *(size_t *)ptr.m_pParams->pUserData;

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mDestruct, IN struct mRefPool_SharedPointerContainer_Internal<T> *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_Success);

  mERROR_CHECK(mSharedPointer_Destroy(&pData->ptr));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mRefPool_Destroy_Internal, IN mRefPool<T> *pRefPool)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRefPool == nullptr, mR_ArgumentNull);

  if (pRefPool->keepForever)
  {
    for (auto _item : pRefPool->ptrs->Iterate())
    {
      mERROR_IF(_item.pData == nullptr || (*_item).ptr == nullptr, mR_ArgumentNull);

      mERROR_IF((*_item).ptr.m_pParams->referenceCount > 1, mR_ResourceStateInvalid); // this lambda also owns it.
      mERROR_CHECK(mSharedPointer_Destroy(&(*_item).ptr));
    }
  }

  mERROR_CHECK(mPool_Destroy(&pRefPool->ptrs));
  mERROR_CHECK(mPool_Destroy(&pRefPool->data));
  mERROR_CHECK(mRecursiveMutex_Destroy(&pRefPool->pMutex));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mRefPoolIterator<T>::mRefPoolIterator(typename mPool<mRefPool_SharedPointerContainer_Internal<T>> *pPool) :
  index(0),
  globalIndex(0),
  blockIndex(0),
  flag(1),
  pData(nullptr),
  pPool(pPool)
{ }

template<typename T>
inline typename mRefPoolIterator<T>::IteratorValue mRefPoolIterator<T>::operator*()
{
  return mRefPoolIterator<T>::IteratorValue(pData->ptr, globalIndex - 1);
}

template<typename T>
inline const typename mRefPoolIterator<T>::IteratorValue mRefPoolIterator<T>::operator*() const
{
  return mRefPoolIterator<T>::IteratorValue(pData->ptr, globalIndex - 1);
}

template<typename T>
inline bool mRefPoolIterator<T>::operator != (const typename mRefPoolIterator<T> &)
{
  for (blockIndex; blockIndex < pPool->size; ++blockIndex)
  {
    for (; index < mBYTES_OF(pPool->pIndexes[0]); ++index)
    {
      if (pPool->pIndexes[blockIndex] & flag)
      {
        const mResult result = mChunkedArray_PointerAt(pPool->data, globalIndex, &pData);

        if (mFAILED(result))
        {
          mFAIL_DEBUG("mChunkedArray_PointerAt failed in mRefPoolIterator::operator != with errorcode %" PRIu64 ".", (uint64_t)result);
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
inline typename mRefPoolIterator<T>& mRefPoolIterator<T>::operator++()
{
  return *this;
}

template<typename T>
inline mRefPoolIterator<T>::IteratorValue::IteratorValue(mPtr<T> &data, const size_t index) :
  data(data),
  index(index)
{ }

template<typename T>
inline T& mRefPoolIterator<T>::IteratorValue::operator*()
{
  return *data;
}

template<typename T>
inline const T& mRefPoolIterator<T>::IteratorValue::operator*() const
{
  return *data;
}
